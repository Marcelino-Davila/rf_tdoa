# TDOA/position_estimator.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import numpy as np

C_MPS = 299_792_458.0  # speed of light (m/s)


def equilateral_triangle_receivers(side_m: float = 1.0) -> Dict[str, np.ndarray]:
    """
    Returns receiver positions for an equilateral triangle of given side length.

    rx0: (0, 0)
    rx1: (L, 0)
    rx2: (L/2, sqrt(3)/2 * L)
    """
    L = float(side_m)
    return {
        "rx0": np.array([0.0, 0.0], dtype=float),
        "rx1": np.array([L, 0.0], dtype=float),
        "rx2": np.array([0.5 * L, (np.sqrt(3.0) / 2.0) * L], dtype=float),
    }


@dataclass
class TDOALocator2D:
    """
    2D TDOA position estimator using Gauss-Newton / Levenberg-Marquardt-ish damping.

    Input:
      tdoa_s: dict like {"rx1": dt10, "rx2": dt20} where dt is seconds relative to ref_rx.

    Output:
      (x, y) in meters, in the same coordinate system as rx_positions.

    Notes:
      - Needs at least 2 non-reference receivers.
      - Returns None if it fails to converge.
    """
    rx_positions: Dict[str, np.ndarray]
    ref_rx: str = "rx0"
    c_mps: float = C_MPS

    def solve(
        self,
        tdoa_s: Dict[str, float],
        x0: Optional[np.ndarray] = None,
        max_iter: int = 50,
        tol: float = 1e-9,
        damping: float = 1e-3,
    ) -> Optional[Tuple[float, float]]:
        if self.ref_rx not in self.rx_positions:
            raise ValueError(f"ref_rx {self.ref_rx} not in rx_positions")

        # Ensure we have at least 2 equations (2D)
        keys = [k for k in tdoa_s.keys() if k != self.ref_rx]
        if len(keys) < 2:
            return None

        r0 = self.rx_positions[self.ref_rx]

        # Initial guess:
        # If none provided, start near the triangle centroid.
        if x0 is None:
            pts = np.stack(list(self.rx_positions.values()), axis=0)
            x = pts.mean(axis=0).copy()
        else:
            x = np.array(x0, dtype=float).copy()

        def dist(p: np.ndarray, r: np.ndarray) -> float:
            return float(np.linalg.norm(p - r))

        for _ in range(max_iter):
            d0 = dist(x, r0)

            # Residual vector f(p) = (di - d0) - c*Δti0
            f = []
            J = []  # Jacobian rows wrt [x, y]

            for rx in keys:
                ri = self.rx_positions.get(rx, None)
                if ri is None:
                    continue

                di = dist(x, ri)
                target = self.c_mps * float(tdoa_s[rx])
                f_i = (di - d0) - target
                f.append(f_i)

                # derivative of distance: ∂/∂p ||p-r|| = (p-r)/||p-r||
                # so row: ( (x-ri)/di - (x-r0)/d0 )
                # guard against divide-by-zero
                if di < 1e-12 or d0 < 1e-12:
                    return None

                grad_i = (x - ri) / di - (x - r0) / d0
                J.append(grad_i)

            f = np.array(f, dtype=float).reshape(-1, 1)      # (m,1)
            J = np.array(J, dtype=float).reshape(-1, 2)      # (m,2)

            # Solve (J^T J + λI) δ = -J^T f
            H = J.T @ J
            g = J.T @ f
            H_damped = H + damping * np.eye(2)

            try:
                delta = np.linalg.solve(H_damped, -g).reshape(2)
            except np.linalg.LinAlgError:
                return None

            x_new = x + delta

            if np.linalg.norm(delta) < tol:
                x = x_new
                break

            x = x_new

        return float(x[0]), float(x[1])
