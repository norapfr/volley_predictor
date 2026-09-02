"""
Red neuronal (MLP) — Fase 8, solo si mejora sobre CatBoost (spec).

A diferencia de los modelos de árboles (Fases 4-7), una red neuronal:
- No maneja NaN de forma nativa — hay que imputar explícitamente. Se usa
  la mediana del tramo de entrenamiento (nunca de test, para no filtrar
  información), y se añade una columna binaria "_was_missing" por cada
  feature que tuviera NaN — así el modelo no pierde la señal de "no había
  historial" que sí aprovechan los árboles automáticamente.
- Necesita las features en una escala comparable (`StandardScaler`), o el
  optimizador converge mal.
- Necesita early stopping explícito: con ~1000-3000 filas de entrenamiento
  por fold, una red sin frenos memoriza en unas pocas épocas. Se reserva un
  tramo cronológico final del propio train (el mismo patrón que en la Fase 5
  para calibración) como validación interna para decidir cuándo parar.

Arquitectura deliberadamente pequeña (2 capas ocultas, dropout) — con este
volumen de datos, una red grande no tiene forma de generalizar mejor que
memorizar.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class NeuralWinModel:
    def __init__(
        self,
        hidden_sizes=(32, 16),
        dropout: float = 0.3,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-3,
        max_epochs: int = 300,
        patience: int = 20,
        val_fraction: float = 0.15,
        date_col: str = "date",
        seed: int = 0,
    ):
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.patience = patience
        self.val_fraction = val_fraction
        self.date_col = date_col
        self.seed = seed

        self._scaler: Optional[StandardScaler] = None
        self._medians: Optional[pd.Series] = None
        self._missing_flag_columns: List[str] = []
        self._model = None  # torch.nn.Module, import perezoso

    def _impute(self, df: pd.DataFrame, feature_columns: List[str], fitting: bool) -> pd.DataFrame:
        X = df[feature_columns].copy()
        if fitting:
            self._medians = X.median()
            # Si una columna sale completamente vacía en el train (p.ej.
            # tournament_progress cuando no hay tournament_no disponible), su
            # mediana también es NaN — rellenar con NaN no rellena nada. Se
            # usa 0.0 como último recurso solo en ese caso extremo.
            self._medians = self._medians.fillna(0.0)
            self._missing_flag_columns = [c for c in feature_columns if X[c].isna().any()]
        for col in self._missing_flag_columns:
            X[f"{col}_was_missing"] = X[col].isna().astype(float)
        X = X.fillna(self._medians)
        return X

    def fit(self, train_df: pd.DataFrame, feature_columns: List[str], winner_col: str = "winner") -> "NeuralWinModel":
        import torch
        from torch import nn

        torch.manual_seed(self.seed)

        ordered = train_df.sort_values(self.date_col, kind="stable")
        split_point = int(len(ordered) * (1 - self.val_fraction))
        fit_part = ordered.iloc[:split_point]
        val_part = ordered.iloc[split_point:]
        if len(val_part) == 0:  # muy pocos datos (p.ej. tests) -> sin early stopping real
            fit_part, val_part = ordered, ordered

        X_fit = self._impute(fit_part, feature_columns, fitting=True)
        self._scaler = StandardScaler().fit(X_fit)
        X_fit_scaled = self._scaler.transform(X_fit)
        y_fit = (fit_part[winner_col] == "team_a").astype(float).to_numpy()

        X_val = self._impute(val_part, feature_columns, fitting=False)
        X_val_scaled = self._scaler.transform(X_val)
        y_val = (val_part[winner_col] == "team_a").astype(float).to_numpy()

        n_features = X_fit_scaled.shape[1]
        layers = []
        prev_size = n_features
        for hidden_size in self.hidden_sizes:
            layers += [nn.Linear(prev_size, hidden_size), nn.ReLU(), nn.Dropout(self.dropout)]
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())
        model = nn.Sequential(*layers)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        loss_fn = nn.BCELoss()

        X_fit_t = torch.tensor(X_fit_scaled, dtype=torch.float32)
        y_fit_t = torch.tensor(y_fit, dtype=torch.float32).unsqueeze(1)
        X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32)
        y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

        best_val_loss = float("inf")
        best_state = None
        epochs_without_improvement = 0

        for _epoch in range(self.max_epochs):
            model.train()
            optimizer.zero_grad()
            preds = model(X_fit_t)
            loss = loss_fn(preds, y_fit_t)
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                val_loss = loss_fn(model(X_val_t), y_val_t).item()

            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        self._model = model
        self._feature_columns = feature_columns
        return self

    def predict_proba_batch(self, test_df: pd.DataFrame, feature_columns: List[str]) -> np.ndarray:
        import torch

        X = self._impute(test_df, feature_columns, fitting=False)
        X_scaled = self._scaler.transform(X)
        X_t = torch.tensor(X_scaled, dtype=torch.float32)
        with torch.no_grad():
            preds = self._model(X_t).squeeze(1).numpy()
        return preds
