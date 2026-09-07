import torch
import torch.nn as nn



class Seq2SeqLSTM(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        output_dim: int,
        pred_len: int,
        dropout: float = 0.0,
        use_reservoir_emb: bool = False,
        num_reservoirs: int = 135,
        reservoir_emb_dim: int = 16,
        use_res_static: bool = False,
        res_static_dim: int = 0,
        use_meta_only_static: bool = False,   # exp4: attributes without the reservoir-ID embedding
        meta_only_static_dim: int = 0,
        latent_mode: str = "attn",          # "last" | "attn"
        use_latent_proj: bool = True,       # LayerNorm + projection head
        # DARSD
        use_darsd: bool = False,
        lcib_k: int = 16,   # number of basis vectors
        emb_dropout_p: float = 0.0,  # reservoir embedding dropout: zeroes embedding with prob p during training
        # forecasting backbone selection: the recurrent encoder-decoder, or a
        # complete Transformer encoder-decoder in its place.
        backbone: str = "lstm",   # "lstm" | "transformer_seq2seq"
        n_heads: int = 8,         # transformer: attention heads (hidden_dim must be divisible)
        tf_layers: int = 2,       # transformer: number of encoder layers
        tf_ff_mult: int = 4,      # transformer: feedforward dim = tf_ff_mult * hidden_dim
        tin: int = 30,            # transformer: input window length (positional-embedding size)
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pred_len = pred_len
        self.use_reservoir_emb = use_reservoir_emb

        # -----------------------
        # Reservoir embedding
        # -----------------------
        if use_reservoir_emb:
            self.reservoir_emb = nn.Embedding(num_reservoirs, reservoir_emb_dim)
            enc_in_dim = input_dim + reservoir_emb_dim
            # cold-start robustness: randomly zero the full embedding vector during training
            self.emb_dropout = nn.Dropout(p=float(emb_dropout_p)) if float(emb_dropout_p) > 0 else None
        else:
            self.reservoir_emb = None
            enc_in_dim = input_dim
            self.emb_dropout = None
            
        self.use_res_static = use_res_static
        self.res_static_dim = int(res_static_dim)

        self.use_meta_only_static = bool(use_meta_only_static)
        self.meta_only_static_dim = int(meta_only_static_dim)

        if self.use_res_static and (not self.use_reservoir_emb):
            raise ValueError("use_res_static=True requires use_reservoir_emb=True.")
        if self.use_res_static and self.res_static_dim <= 0:
            raise ValueError("use_res_static=True but res_static_dim<=0.")

        if self.use_meta_only_static and self.use_reservoir_emb:
            raise ValueError(
                "use_meta_only_static=True requires use_reservoir_emb=False "
                "for a clean metadata+pure_lstm experiment."
            )
        if self.use_meta_only_static and self.use_res_static:
            raise ValueError("use_meta_only_static=True requires use_res_static=False.")
        if self.use_meta_only_static and self.meta_only_static_dim <= 0:
            raise ValueError("use_meta_only_static=True but meta_only_static_dim<=0.")

        # determine active metadata dimensionality
        if self.use_res_static:
            self.metadata_dim = int(self.res_static_dim)
        elif self.use_meta_only_static:
            self.metadata_dim = int(self.meta_only_static_dim)
        else:
            self.metadata_dim = 0

        if self.metadata_dim > 0:
            # one shared attribute buffer; train/eval can still call old setter names
            self.register_buffer(
                "metadata_static",
                torch.zeros(num_reservoirs, self.metadata_dim),
                persistent=False,
            )

            # the attribute branches are constructed LATER, after the backbone modules,
            # so that enabling them does not change the backbone's random initialisation.
            self.meta_to_hidden = None
            self.meta_hidden_gate = None
        else:
            self.metadata_static = None

            self.meta_to_hidden = None
            self.meta_hidden_gate = None

        # input encoder
        self.input_encoder = nn.Sequential(
            nn.Linear(enc_in_dim, hidden_dim),
            nn.ReLU(),
        )

        # encoder / decoder
        self.backbone = str(backbone).lower().strip()
        if self.backbone not in ("lstm", "transformer_seq2seq"):
            raise ValueError(
                f"Unsupported backbone={backbone!r}; choose 'lstm' or 'transformer_seq2seq'."
            )

        if self.backbone == "lstm":
            self.encoder = nn.LSTM(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.decoder = None
            self.pos_src = None
        else:
            # Full Transformer forecasting backbone. The encoder models the
            # hydrometeorological history, while the decoder receives the RSSD-
            # decomposed and static-conditioned state as its forecast query and
            # attends to the complete encoded history for all lead days.
            self.n_heads = int(n_heads)
            self.tf_layers = int(tf_layers)
            self.tf_ff_mult = int(tf_ff_mult)
            self.tin = int(tin)
            if hidden_dim % self.n_heads != 0:
                raise ValueError(
                    f"hidden_dim={hidden_dim} must be divisible by n_heads={self.n_heads}."
                )
            enc_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=self.n_heads,
                dim_feedforward=hidden_dim * self.tf_ff_mult,
                dropout=dropout,
                batch_first=True,
            )
            dec_layer = nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=self.n_heads,
                dim_feedforward=hidden_dim * self.tf_ff_mult,
                dropout=dropout,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=self.tf_layers)
            self.decoder = nn.TransformerDecoder(dec_layer, num_layers=self.tf_layers)
            self.pos_src = nn.Embedding(self.tin, hidden_dim)
            self.pos_tgt = nn.Embedding(self.pred_len, hidden_dim)

        # Forecast head. Every lead day is produced in one pass: the recurrent backbone
        # maps the conditioned state straight to all of them, and the Transformer backbone
        # applies the same head to each of its lead-day decoder outputs.
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.output_dim = output_dim
        if self.backbone == "transformer_seq2seq":
            self.fc2 = nn.Linear(hidden_dim, output_dim)
            self.fc2_direct = None
        else:
            self.fc2 = None
            self.fc2_direct = nn.Linear(hidden_dim, output_dim * pred_len)
        self.inp_dropout = nn.Dropout(dropout)
        self.mlp_dropout = nn.Dropout(dropout)

        # -------------------------------------------------
        # build metadata branches AFTER backbone init
        # so exp4 does not change the random initialization
        # of the pure_lstm backbone under the same seed.
        # -------------------------------------------------
        def _make_scalar_branch(
            out_dim: int,
            hidden_dim_branch: int,
            last_weight_std: float = 1e-3,
            zero_last_bias: bool = True,
        ):
            seq = nn.Sequential(
                nn.Linear(1, hidden_dim_branch),
                nn.Tanh(),
                nn.Linear(hidden_dim_branch, out_dim),
            )
            if (last_weight_std is None) or (float(last_weight_std) <= 0):
                nn.init.zeros_(seq[-1].weight)
            else:
                nn.init.normal_(seq[-1].weight, mean=0.0, std=float(last_weight_std))
            if zero_last_bias:
                nn.init.zeros_(seq[-1].bias)
            return seq

        if self.metadata_dim > 0:
            # One attribute pathway, shared by the attribute-only variant and the full model:
            # each standardised attribute is mapped to an H-dimensional vector by its own
            # branch, and their sum conditions the state that reaches the forecast head.

            meta_hidden_branch = max(16, hidden_dim // 4)

            # exp4 can start slightly stronger because it has no reservoir embedding branch;
            # full model keeps a milder init to avoid overwhelming the dynamic trunk.
            hidden_last_std = 1e-2 if self.use_meta_only_static else 5e-3

            self.meta_to_hidden = nn.ModuleList([
                _make_scalar_branch(hidden_dim, meta_hidden_branch, last_weight_std=hidden_last_std)
                for _ in range(self.metadata_dim)
            ])
            self.meta_hidden_gate = nn.Parameter(torch.tensor(0.0))

        # dynamic-state readout
        if latent_mode not in ("last", "attn"):
            raise ValueError(f"[Seq2SeqLSTM] Unsupported latent_mode={latent_mode}")
        self.latent_mode = latent_mode

        # attention pooling module (only used when latent_mode=="attn")
        attn_hidden = max(16, hidden_dim // 2)
        self.latent_attn = nn.Sequential(
            nn.Linear(hidden_dim, attn_hidden),
            nn.Tanh(),
            nn.Linear(attn_hidden, 1),
        )

        # LayerNorm + projection head
        self.use_latent_proj = use_latent_proj
        if use_latent_proj:
            self.latent_ln = nn.LayerNorm(hidden_dim)
            self.latent_proj = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        else:
            self.latent_ln = None
            self.latent_proj = None
        
        # DARSD: Adv-LCIB
        self.use_darsd = bool(use_darsd)
        self.lcib_k = int(lcib_k)
        if self.use_darsd:
            if self.lcib_k <= 1 or self.lcib_k > hidden_dim:
                raise ValueError(f"use_darsd=True but lcib_k invalid: {self.lcib_k} (hidden_dim={hidden_dim})")
            # learnable basis B \in R^{H x K}
            self.lcib_B = nn.Parameter(torch.empty(hidden_dim, self.lcib_k))
            nn.init.orthogonal_(self.lcib_B)  # good init
            # gate to blend: h = (1-g)*h_raw + g*h_lcib  (prevents collapse)
            self.lcib_gate = nn.Parameter(torch.tensor(0.0))  # sigmoid(0)=0.5
            # temperature for softmax (can tune later; keep stable default)
            self.lcib_tau = 1.0
            # cache for training-time regularizers (filled in forward)
            self._lcib_last_w = None
            self._lcib_last_coeffs = None
        else:
            self.lcib_B = None
            self.lcib_gate = None
            self.lcib_tau = 1.0
            self._lcib_last_w = None
            self._lcib_last_coeffs = None

    def set_res_static(self, res_static: torch.Tensor):
        """
        res_static: (num_reservoirs, metadata_dim), aligned with node-order reservoir_ids.
        Used by full-model experiments (exp2 / exp3) under the new unified metadata pathway.
        """
        if not self.use_res_static:
            raise RuntimeError("set_res_static called but use_res_static=False.")
        if not torch.is_tensor(res_static):
            raise TypeError("res_static must be a torch.Tensor.")
        if res_static.dim() != 2:
            raise ValueError(f"res_static must be 2D, got shape={tuple(res_static.shape)}")
        if tuple(res_static.shape) != tuple(self.metadata_static.shape):
            raise ValueError(
                f"res_static shape mismatch. Expected {tuple(self.metadata_static.shape)}, got {tuple(res_static.shape)}"
            )
        self.metadata_static.copy_(res_static.detach().to(self.metadata_static.device, dtype=self.metadata_static.dtype))

    def set_meta_static(self, meta_static: torch.Tensor):
        """
        meta_static: (num_reservoirs, metadata_dim), aligned with node-order reservoir_ids.
        Used by exp4 metadata+pure_lstm under the new unified metadata pathway.
        """
        if not self.use_meta_only_static:
            raise RuntimeError("set_meta_static called but use_meta_only_static=False.")
        if not torch.is_tensor(meta_static):
            raise TypeError("meta_static must be a torch.Tensor.")
        if meta_static.dim() != 2:
            raise ValueError(f"meta_static must be 2D, got shape={tuple(meta_static.shape)}")
        if tuple(meta_static.shape) != tuple(self.metadata_static.shape):
            raise ValueError(
                f"meta_static shape mismatch. Expected {tuple(self.metadata_static.shape)}, got {tuple(meta_static.shape)}"
            )
        self.metadata_static.copy_(meta_static.detach().to(self.metadata_static.device, dtype=self.metadata_static.dtype))

    def _metadata_branch_sum(self, meta: torch.Tensor, branches: nn.ModuleList):
        """
        meta: (B, metadata_dim)
        branches: one independent branch per metadata scalar
        returns: (B, out_dim)
        """
        if branches is None:
            return None
        if meta.dim() != 2:
            raise RuntimeError(f"_metadata_branch_sum expects 2D meta, got {tuple(meta.shape)}")
        if int(meta.size(1)) != int(self.metadata_dim):
            raise RuntimeError(
                f"_metadata_branch_sum metadata_dim mismatch: got {int(meta.size(1))}, expected {int(self.metadata_dim)}"
            )

        out = None
        for i, branch in enumerate(branches):
            contrib = branch(meta[:, i:i+1])
            out = contrib if out is None else (out + contrib)
        return out

    def _metadata_to_hidden(self, meta: torch.Tensor):
        return self._metadata_branch_sum(meta, self.meta_to_hidden)

    def _resolve_meta_for_batch(
        self,
        total_nodes: int,
        device,
        reservoir_ids: torch.Tensor = None,
    ):
        if not (self.use_meta_only_static or self.use_res_static):
            return None, reservoir_ids

        if self.metadata_static is None:
            raise RuntimeError("metadata pathway enabled but metadata_static is None.")

        base_nodes = int(self.metadata_static.shape[0])

        if reservoir_ids is None:
            if total_nodes % base_nodes != 0:
                raise RuntimeError(
                    f"metadata fallback reservoir_ids build failed: total_nodes={total_nodes} "
                    f"is not divisible by num_meta_rows={base_nodes}"
                )
            repeat_factor = total_nodes // base_nodes
            reservoir_ids = torch.arange(base_nodes, device=device, dtype=torch.long).repeat(repeat_factor)
        else:
            reservoir_ids = reservoir_ids.to(device=device, dtype=torch.long)
            if reservoir_ids.numel() != total_nodes:
                raise RuntimeError(
                    f"reservoir_ids length {reservoir_ids.numel()} != nodes {total_nodes}."
                )

        min_id = int(reservoir_ids.min().item())
        max_id = int(reservoir_ids.max().item())
        if (min_id < 0) or (max_id >= base_nodes):
            raise RuntimeError(
                f"metadata reservoir_ids out of range: min={min_id} max={max_id} "
                f"num_meta_rows={base_nodes}"
            )

        meta = self.metadata_static[reservoir_ids]
        return meta, reservoir_ids

    def _pool_latent(self, encoder_outputs: torch.Tensor) -> torch.Tensor:
        """
        encoder_outputs: (B, T, H)
        returns h_latent: (B, H)
        """
        if self.latent_mode == "last":
            h_latent = encoder_outputs[:, -1, :]
        else:  # "attn"
            scores = self.latent_attn(encoder_outputs).squeeze(-1)  # (B, T)
            w = torch.softmax(scores, dim=1).unsqueeze(-1)          # (B, T, 1)
            h_latent = torch.sum(encoder_outputs * w, dim=1)        # (B, H)
        return h_latent

    def _project_latent(self, h_latent: torch.Tensor) -> torch.Tensor:
        if not self.use_latent_proj:
            return h_latent
        h = self.latent_ln(h_latent)
        h = self.latent_proj(h)
        return h

    def _lcib_decompose(self, h: torch.Tensor):
        """Decompose the dynamic state into a shared and a site-specific component.

        The basis weights are a softmax over the inner products between the dynamic
        state and the shared basis; the shared component is the weighted combination
        of basis vectors they select, and the site-specific component is what that
        reconstruction leaves behind.
        """
        if (not self.use_darsd) or (self.lcib_B is None):
            raise RuntimeError("_lcib_decompose requires use_darsd=True.")

        Bmat = self.lcib_B  # (H, K)
        coordinates = torch.softmax(
            (h @ Bmat) / max(self.lcib_tau, 1e-6),
            dim=-1,
        )
        h_shared = coordinates @ Bmat.T

        h_residual = h - h_shared
        return h_shared, h_residual, coordinates

    def _lcib_forward(self, h: torch.Tensor) -> torch.Tensor:
        """Blend the dynamic state with its DARSD shared component."""
        h_shared, _h_residual, coordinates = self._lcib_decompose(h)

        # blend to avoid hurting baseline
        g = torch.sigmoid(self.lcib_gate)  # scalar
        h_mix = (1.0 - g) * h + g * h_shared

        self._lcib_last_w = coordinates
        self._lcib_last_coeffs = coordinates
        return h_mix

    def darsd_regularizer(self, entropy_weight: float = 0.01, eps: float = 1e-8) -> torch.Tensor:
        """
        returns a scalar regularizer:
        - orthogonality of basis B
        - low-entropy (peaky) selection over basis (softmax regularization)
        """
        if (not getattr(self, "use_darsd", False)) or (self.lcib_B is None):
            return torch.tensor(0.0, device=next(self.parameters()).device)

        Bmat = self.lcib_B  # (H, K)
        K = Bmat.shape[1]
        I = torch.eye(K, device=Bmat.device, dtype=Bmat.dtype)
        BtB = Bmat.T @ Bmat
        orth_loss = (BtB - I).pow(2).mean()

        if self._lcib_last_w is None:
            return orth_loss
        w = self._lcib_last_w  # (B, K)
        entropy = -(w * torch.log(w.clamp_min(eps))).sum(dim=-1).mean()  # positive
        return orth_loss + entropy_weight * entropy

    @staticmethod
    def _stack_time_graphs(graph_list):
        # graph_list: length Tin, each g.x is (nodes, F) or (nodes, 1, F)
        xs = []
        for g in graph_list:
            xg = g.x
            if xg.dim() == 3 and xg.size(1) == 1:
                xg = xg[:, 0, :]  # (nodes, F)
            if xg.dim() != 2:
                raise RuntimeError(f"Unexpected g.x shape: {tuple(xg.shape)}")
            xs.append(xg)
        x = torch.stack(xs, dim=0)          # (Tin, nodes, F)
        x = x.permute(1, 0, 2).contiguous() # (nodes, Tin, F)
        return x

    def _build_encoded_sequence(
        self,
        graph_list,
        reservoir_ids: torch.Tensor = None,
        use_domain_cond: bool = True,
    ):
        x = self._stack_time_graphs(graph_list)

        if self.use_reservoir_emb:
            if use_domain_cond:
                if reservoir_ids is None:
                    reservoir_ids = torch.arange(x.size(0), device=x.device, dtype=torch.long)
                else:
                    if reservoir_ids.numel() != x.size(0):
                        raise RuntimeError(
                            f"reservoir_ids length {reservoir_ids.numel()} != nodes {x.size(0)}."
                        )

                min_id = int(reservoir_ids.min().item())
                max_id = int(reservoir_ids.max().item())
                if (min_id < 0) or (max_id >= self.reservoir_emb.num_embeddings):
                    raise RuntimeError(
                        f"reservoir_ids out of range: min={min_id} max={max_id} "
                        f"num_embeddings={self.reservoir_emb.num_embeddings}."
                    )

                emb = self.reservoir_emb(reservoir_ids)
                # embedding dropout: zero entire embedding vector with prob p during training
                # forces model to function without embeddings → better cold-start generalization
                if self.training and (self.emb_dropout is not None):
                    mask = (torch.rand(emb.size(0), 1, device=emb.device) > self.emb_dropout.p).float()
                    emb = emb * mask
            else:
                # MMD/domain-agnostic path:
                # keep encoder input dimensionality consistent with full model,
                # but do not inject reservoir-specific information.
                emb = torch.zeros(
                    x.size(0),
                    self.reservoir_emb.embedding_dim,
                    device=x.device,
                    dtype=x.dtype,
                )

            emb_seq = emb.unsqueeze(1).expand(-1, x.size(1), -1)
            x = torch.cat([x, emb_seq], dim=-1)

        x = self.input_encoder(x)
        x = self.inp_dropout(x)

        # Reservoir attributes are not injected here: they condition the state after the
        # decomposition, so the RSSD layer and the alignment losses both act on a purely
        # dynamic state.
        if self.backbone == "lstm":
            encoder_outputs, _state = self.encoder(x)
        else:
            # transformer backbone: add learnable positional encoding, then self-attention.
            T = x.size(1)
            x = x + self.pos_src.weight[:T].unsqueeze(0)   # (1, T, H) broadcast over batch
            encoder_outputs = self.encoder(x)              # (B, T, H)
        return encoder_outputs

    def encode_latent(
        self,
        graph_list,
        reservoir_ids: torch.Tensor = None,
        use_domain_cond: bool = True,
        apply_darsd: bool = None,
        return_sequence: bool = False,
    ):
        encoder_outputs = self._build_encoded_sequence(
            graph_list,
            reservoir_ids=reservoir_ids,
            use_domain_cond=use_domain_cond,
        )

        h_latent = self._pool_latent(encoder_outputs)
        h_latent = self._project_latent(h_latent)

        if apply_darsd is None:
            apply_darsd = bool(self.use_darsd)

        if apply_darsd:
            h_latent = self._lcib_forward(h_latent)

        if (self.metadata_dim > 0) and use_domain_cond and (self.metadata_static is not None):
            meta, reservoir_ids = self._resolve_meta_for_batch(
                total_nodes=int(h_latent.size(0)),
                device=h_latent.device,
                reservoir_ids=reservoir_ids,
            )
            meta_h = self._metadata_to_hidden(meta)  # (B, H)
            gate = torch.sigmoid(self.meta_hidden_gate) if (self.meta_hidden_gate is not None) else 1.0

            # unified metadata path:
            # - exp2: DARSD first, then metadata conditions the forecast state
            # - exp3: MMD uses use_domain_cond=False, so metadata stays outside alignment loss
            # - exp4: same hidden-state conditioning is reused without reservoir embeddings
            h_latent = h_latent + gate * meta_h

        if return_sequence:
            return h_latent, encoder_outputs
        return h_latent

    def forward(
        self,
        graph_list,
        return_latent: bool = False,
        reservoir_ids: torch.Tensor = None,
    ):
        h_latent, encoder_outputs = self.encode_latent(
            graph_list,
            reservoir_ids=reservoir_ids,
            use_domain_cond=True,
            apply_darsd=bool(self.use_darsd),
            return_sequence=True,
        )

        # ---- full Transformer encoder--decoder forecasting backbone ----
        if self.backbone == "transformer_seq2seq":
            batch_size = h_latent.size(0)
            lead_queries = self.pos_tgt.weight[: self.pred_len].unsqueeze(0)
            lead_queries = lead_queries.expand(batch_size, -1, -1)
            decoder_input = lead_queries + h_latent.unsqueeze(1)
            causal_mask = nn.Transformer.generate_square_subsequent_mask(
                self.pred_len,
                device=decoder_input.device,
                dtype=decoder_input.dtype,
            )
            decoder_output = self.decoder(
                tgt=decoder_input,
                memory=encoder_outputs,
                tgt_mask=causal_mask,
            )
            h = self.relu(self.fc1(decoder_output))
            h = self.mlp_dropout(h)
            y_hat = self.fc2(h)

            if return_latent:
                return y_hat, h_latent
            return y_hat

        # ---- recurrent backbone: every lead day from the conditioned state in one pass ----
        h = self.relu(self.fc1(h_latent))                            # (B, H)
        h = self.mlp_dropout(h)
        out = self.fc2_direct(h)                                     # (B, pred_len*output_dim)
        y_hat = out.view(h.size(0), self.pred_len, self.output_dim)  # (B, pred_len, output_dim)

        if return_latent:
            return y_hat, h_latent
        return y_hat
