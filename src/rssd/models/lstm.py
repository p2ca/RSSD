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
        use_direct_head: bool = False,   # B2: direct multi-horizon head (non-autoregressive)
        dropout: float = 0.0,
        use_reservoir_emb: bool = False,
        num_reservoirs: int = 135,
        reservoir_emb_dim: int = 16,
        use_res_static: bool = False,      # Step2.3
        res_static_dim: int = 0,           # Step2.3
        res_static_mode: str = "latent",   # legacy config key; v16+ uses latent-stage metadata injection
        film_gamma_scale: float = 0.10,    # FiLM gamma amplitude (safety)
        film_beta_scale: float = 0.10,     # FiLM beta amplitude (safety)
        use_meta_only_static: bool = False,   # exp4: metadata + pure_lstm
        meta_only_static_dim: int = 0,        # fixed to 5 in this project
        meta_feature_strengths = None,       # length = meta_only_static_dim, 0 means no injection for that metadata
        latent_mode: str = "attn",          # "last" | "mean" | "attn" | "last_mean"
        use_latent_proj: bool = True,       # A2: LayerNorm + projection head
        use_err_head: bool = False,         # C1: latent->error/difficulty head
        err_head_hidden: int = 0,           # 0 => auto
        # DARSD
        use_darsd: bool = False,
        lcib_k: int = 16,   # number of basis vectors
        darsd_mode: str = "softmax_reconstruction",
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
            
        # unified metadata conditioning (replaces old res_static_proj / res_static_film / meta_static_proj)
        self.use_res_static = use_res_static
        self.res_static_dim = int(res_static_dim)

        self.res_static_mode = str(res_static_mode).lower().strip()
        self.film_gamma_scale = float(film_gamma_scale)
        self.film_beta_scale = float(film_beta_scale)

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
            if meta_feature_strengths is None:
                meta_feature_strengths = [1.0] * self.metadata_dim
            meta_feature_strengths = [float(v) for v in meta_feature_strengths]
            if len(meta_feature_strengths) != self.metadata_dim:
                raise ValueError(
                    f"meta_feature_strengths length mismatch: "
                    f"expected {self.metadata_dim}, got {len(meta_feature_strengths)}"
                )

            # one shared metadata buffer; train/eval can still call old setter names
            self.register_buffer(
                "metadata_static",
                torch.zeros(num_reservoirs, self.metadata_dim),
                persistent=False,
            )
            self.register_buffer(
                "meta_feature_strengths",
                torch.tensor(meta_feature_strengths, dtype=torch.float32),
                persistent=False,
            )

            # backward-compatible aliases for shape/debug only; do NOT use these for runtime indexing
            self.res_static = self.metadata_static if self.use_res_static else None
            self.meta_static = self.metadata_static if self.use_meta_only_static else None

            # metadata modules will be constructed LATER,
            # after core backbone modules are created, to avoid changing backbone init.
            self.meta_to_emb = None
            self.meta_to_film = None
            self.meta_to_hidden = None
            self.meta_hidden_gate = None

            self.res_static_gate = None
            self.res_static_proj = None
            self.res_static_film = None
            self.meta_static_proj = None
        else:
            self.metadata_static = None
            self.meta_feature_strengths = None

            self.res_static = None
            self.meta_static = None

            self.meta_to_emb = None
            self.meta_to_film = None
            self.meta_to_hidden = None
            self.meta_hidden_gate = None

            self.res_static_gate = None

            self.res_static_proj = None
            self.res_static_film = None
            self.meta_static_proj = None

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
            self.decoder = nn.LSTM(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
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

        # output head
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.output_dim = output_dim
        self.use_direct_head = bool(use_direct_head)
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
            # v16+: one unified metadata pathway for both:
            # - exp4 metadata + pure_lstm
            # - exp2 / exp3 full model
            #
            # Metadata is converted into a latent-state correction instead of being injected
            # into reservoir embedding / encoder FiLM. This keeps domain adaptation focused on
            # dynamic latent, while allowing metadata to condition the final forecast state.

            meta_hidden_branch = max(16, hidden_dim // 4)

            # exp4 can start slightly stronger because it has no reservoir embedding branch;
            # full model keeps a milder init to avoid overwhelming the dynamic trunk.
            hidden_last_std = 1e-2 if self.use_meta_only_static else 5e-3

            self.meta_to_hidden = nn.ModuleList([
                _make_scalar_branch(hidden_dim, meta_hidden_branch, last_weight_std=hidden_last_std)
                for _ in range(self.metadata_dim)
            ])
            self.meta_hidden_gate = nn.Parameter(torch.tensor(0.0))

            # keep legacy attributes as explicit no-op placeholders for readability / backward inspection
            self.meta_to_emb = None
            self.meta_to_film = None
            self.res_static_gate = None

        # Latent config (A1/A2)
        if latent_mode not in ("last", "mean", "attn", "last_mean"):
            raise ValueError(f"[Seq2SeqLSTM] Unsupported latent_mode={latent_mode}")
        self.latent_mode = latent_mode

        # A1: attention pooling module (only used when latent_mode=="attn")
        attn_hidden = max(16, hidden_dim // 2)
        self.latent_attn = nn.Sequential(
            nn.Linear(hidden_dim, attn_hidden),
            nn.Tanh(),
            nn.Linear(attn_hidden, 1),
        )

        # A2: LayerNorm + projection head
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
        self.darsd_mode = str(darsd_mode).lower().strip()
        valid_darsd_modes = {"softmax_reconstruction", "orthogonal_projection"}
        if self.darsd_mode not in valid_darsd_modes:
            raise ValueError(
                f"Unsupported darsd_mode={darsd_mode!r}; "
                f"choose one of {sorted(valid_darsd_modes)}."
            )
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

        # Latent -> error/difficulty head (C1)
        self.use_err_head = use_err_head
        if use_err_head:
            eh = err_head_hidden if err_head_hidden and err_head_hidden > 0 else max(32, hidden_dim // 2)
            self.err_head = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, eh),
                nn.ReLU(),
                nn.Linear(eh, 1),
            )
        else:
            self.err_head = None

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

        strengths = self.meta_feature_strengths.to(device=meta.device, dtype=meta.dtype)
        out = None
        for i, branch in enumerate(branches):
            contrib = branch(meta[:, i:i+1]) * strengths[i]
            out = contrib if out is None else (out + contrib)
        return out

    def _metadata_to_emb(self, meta: torch.Tensor):
        return self._metadata_branch_sum(meta, self.meta_to_emb)

    def _metadata_to_film(self, meta: torch.Tensor):
        return self._metadata_branch_sum(meta, self.meta_to_film)

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
        elif self.latent_mode == "mean":
            h_latent = encoder_outputs.mean(dim=1)
        elif self.latent_mode == "last_mean":
            h_latent = 0.5 * (encoder_outputs[:, -1, :] + encoder_outputs.mean(dim=1))
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
        """Return DARSD shared state, residual state, and coordinates.

        ``softmax_reconstruction`` preserves the historical checkpoint
        behaviour. ``orthogonal_projection`` defines the shared component as
        the exact projection onto span(B), yielding an additive orthogonal
        shared/residual decomposition.
        """
        if (not self.use_darsd) or (self.lcib_B is None):
            raise RuntimeError("_lcib_decompose requires use_darsd=True.")

        Bmat = self.lcib_B  # (H, K)
        if self.darsd_mode == "softmax_reconstruction":
            coordinates = torch.softmax(
                (h @ Bmat) / max(self.lcib_tau, 1e-6),
                dim=-1,
            )
            h_shared = coordinates @ Bmat.T
        else:
            basis_q, _ = torch.linalg.qr(Bmat, mode="reduced")
            coordinates = h @ basis_q
            h_shared = coordinates @ basis_q.T

        h_residual = h - h_shared
        return h_shared, h_residual, coordinates

    def _lcib_forward(self, h: torch.Tensor) -> torch.Tensor:
        """Blend the dynamic state with its DARSD shared component."""
        h_shared, _h_residual, coordinates = self._lcib_decompose(h)

        # blend to avoid hurting baseline
        g = torch.sigmoid(self.lcib_gate)  # scalar
        h_mix = (1.0 - g) * h + g * h_shared

        # Probability weights exist only in the historical softmax mode.
        self._lcib_last_w = (
            coordinates if self.darsd_mode == "softmax_reconstruction" else None
        )
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

        if self.darsd_mode == "orthogonal_projection":
            return orth_loss

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

        # v16+: no metadata injection at encoder-input stage.
        # Metadata is injected later on latent / decoder state so that:
        # - exp2: DARSD acts on dynamic latent first
        # - exp3: MMD stays on pre-metadata latent
        # - exp4 shares the same hidden-state conditioning idea
        if self.backbone == "lstm":
            encoder_outputs, (h_n, c_n) = self.encoder(x)
        else:
            # transformer backbone: add learnable positional encoding, then self-attention.
            # No recurrent state -> downstream metadata/decoder paths handle h_n/c_n=None.
            T = x.size(1)
            x = x + self.pos_src.weight[:T].unsqueeze(0)   # (1, T, H) broadcast over batch
            encoder_outputs = self.encoder(x)              # (B, T, H)
            h_n, c_n = None, None
        return encoder_outputs, h_n, c_n

    def encode_latent(
        self,
        graph_list,
        reservoir_ids: torch.Tensor = None,
        use_domain_cond: bool = True,
        apply_darsd: bool = None,
        return_state: bool = False,
        return_sequence: bool = False,
    ):
        encoder_outputs, h_n, c_n = self._build_encoded_sequence(
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

            if h_n is not None:
                h_n = h_n.clone()
                h_n[-1] = h_n[-1] + gate * meta_h

        if return_state and return_sequence:
            return h_latent, h_n, c_n, encoder_outputs
        if return_state:
            return h_latent, h_n, c_n
        if return_sequence:
            return h_latent, encoder_outputs
        return h_latent

    def forward(
        self,
        graph_list,
        return_latent: bool = False,
        return_err: bool = False,
        reservoir_ids: torch.Tensor = None,
    ):
        h_latent, h_n, c_n, encoder_outputs = self.encode_latent(
            graph_list,
            reservoir_ids=reservoir_ids,
            use_domain_cond=True,
            apply_darsd=bool(self.use_darsd),
            return_state=True,
            return_sequence=True,
        )

        err_pred = None
        if self.err_head is not None:
            err_pred = self.err_head(h_latent).squeeze(-1)              # (B,)

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

            if return_latent and return_err:
                return y_hat, h_latent, err_pred
            if return_latent:
                return y_hat, h_latent
            if return_err:
                return y_hat, err_pred
            return y_hat

        # ---- B2: direct multi-horizon head (no autoregressive rollout) ----
        if getattr(self, "use_direct_head", False):
            h = self.relu(self.fc1(h_latent))                           # (B, H)
            h = self.mlp_dropout(h)
            out = self.fc2_direct(h)                                    # (B, pred_len*output_dim)
            y_hat = out.view(h.size(0), self.pred_len, self.output_dim) # (B, pred_len, output_dim)

            if return_latent and return_err:
                return y_hat, h_latent, err_pred
            if return_latent:
                return y_hat, h_latent
            if return_err:
                return y_hat, err_pred
            return y_hat
        
        # decoder uses latent as initial input token
        decoder_input = h_latent.unsqueeze(1)                           # (B, 1, H)
        decoder_hidden, decoder_cell = h_n, c_n

        outputs = []
        for t in range(self.pred_len):
            decoder_output, (decoder_hidden, decoder_cell) = self.decoder(
                decoder_input, (decoder_hidden, decoder_cell)
            )                                                           # (B, 1, H)
            h = self.relu(self.fc1(decoder_output))
            h = self.mlp_dropout(h)
            out = self.fc2(h)
            outputs.append(out)
            decoder_input = decoder_output
            
            # ---- (scheme2) stabilize long-horizon rollout ----
            t_fac = float(t) / float(max(self.pred_len - 1, 1))

            anchor_alpha = float(getattr(self, "decoder_anchor_alpha", 0.0))
            if anchor_alpha > 0.0:
                # convex pull-back: decoder_input <- (1-a)*decoder_output + a*h_latent
                a = float(t_fac) * float(anchor_alpha)
                if a < 0.0:
                    a = 0.0
                elif a > 1.0:
                    a = 1.0
                decoder_input = (1.0 - a) * decoder_output + a * h_latent.unsqueeze(1)

            noise_std = float(getattr(self, "decoder_noise_std", 0.0))
            if self.training and (noise_std > 0.0):
                decoder_input = decoder_input + (t_fac * noise_std) * torch.randn_like(decoder_input)            

        y_hat = torch.cat(outputs, dim=1)                                # (B, pred_len, output_dim)

        if return_latent and return_err:
            return y_hat, h_latent, err_pred
        if return_latent:
            return y_hat, h_latent
        if return_err:
            return y_hat, err_pred
        return y_hat


# Autoregressive Decoding: Error propagation ->  Prediction for day 2 depends on day 1 prediction
