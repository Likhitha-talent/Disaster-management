"""
AI Inference & Post-Processing Engine
Performs end-to-end building localization, bi-temporal Siamese damage classification,
footprint polygon aggregation, confidence estimation, and review flagging.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple

import cv2


DAMAGE_CLASSES = {
    0: "undamaged",
    1: "minor_damage",
    2: "major_damage",
    3: "destroyed",
}

DAMAGE_COLORS = {
    "undamaged": "#10B981",    # Emerald Green
    "minor_damage": "#FBBF24",  # Amber/Yellow
    "major_damage": "#F97316",  # Deep Orange
    "destroyed": "#EF4444",     # Crimson Red
}


class DamageInferenceEngine:
    def __init__(
        self,
        confidence_threshold: float = 0.65,
        model_version: str = "Siamese-UNet-v1.0.0",
        device: Optional[str] = None,
    ):
        self.confidence_threshold = confidence_threshold
        self.model_version = model_version
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

    def aggregate_building_damage(
        self,
        damage_prob_map: np.ndarray,
        building_mask: np.ndarray,
        affine_transform: Optional[Tuple[float, float, float, float, float, float]] = None,
        origin_lat_lon: Tuple[float, float] = (12.3051, 76.6551), # Mysore default coordinate anchor
        meters_per_pixel: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """
        Extracts individual building polygons from mask, aggregates Siamese damage pixel probabilities,
        computes calibrated confidence score, and flags uncertain predictions for human review.

        Args:
          damage_prob_map: (4, H, W) numpy array of softmax probabilities for each damage class.
          building_mask: (H, W) binary mask where 1 indicates building footprint.
          origin_lat_lon: Anchor latitude and longitude for georeferencing.
          meters_per_pixel: Ground Sampling Distance (GSD).

        Returns:
          List of GeoJSON Feature dictionaries.
        """
        H, W = building_mask.shape
        building_features = []

        # Find connected components / contours for building polygons
        contours, _ = cv2.findContours(
            building_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        lat0, lon0 = origin_lat_lon
        deg_per_meter_lat = 1.0 / 111320.0
        deg_per_meter_lon = 1.0 / (111320.0 * np.cos(np.radians(lat0)))

        building_id_counter = 1

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 12:  # Ignore noise contours smaller than 3 m²
                continue

            # Create mask for this specific building footprint
            single_bldg_mask = np.zeros((H, W), dtype=np.uint8)
            cv2.drawContours(single_bldg_mask, [cnt], -1, 1, thickness=-1)

            # Extract damage probabilities for pixels inside the building footprint
            building_pixels_mask = single_bldg_mask == 1
            if not np.any(building_pixels_mask):
                continue

            # Mean probabilities for [undamaged, minor, major, destroyed]
            class_probs = damage_prob_map[:, building_pixels_mask].mean(axis=1)

            # Majority class
            pred_class_id = int(np.argmax(class_probs))
            pred_class_name = DAMAGE_CLASSES[pred_class_id]
            confidence = float(class_probs[pred_class_id])

            # Review flag for human-in-the-loop audit
            needs_review = confidence < self.confidence_threshold
            review_status = "FLAGGED_FOR_REVIEW" if needs_review else "AUTO_VERIFIED"

            # Convert contour pixel coordinates to geographic coordinates (WGS84 lon, lat)
            geo_coords = []
            for pt in cnt[:, 0, :]:
                px, py = pt[0], pt[1]
                # Center around anchor
                dx = (px - W / 2) * meters_per_pixel
                dy = (H / 2 - py) * meters_per_pixel

                lon = lon0 + dx * deg_per_meter_lon
                lat = lat0 + dy * deg_per_meter_lat
                geo_coords.append([round(lon, 7), round(lat, 7)])

            # Close polygon
            if len(geo_coords) >= 3:
                geo_coords.append(geo_coords[0])

                feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [geo_coords]
                    },
                    "properties": {
                        "building_id": f"BLD-{building_id_counter:04d}",
                        "damage_class": pred_class_name,
                        "damage_level": pred_class_id,
                        "damage_color": DAMAGE_COLORS[pred_class_name],
                        "confidence": round(confidence, 4),
                        "review_status": review_status,
                        "needs_review": needs_review,
                        "area_sqm": round(area * (meters_per_pixel ** 2), 1),
                        "class_probabilities": {
                            DAMAGE_CLASSES[i]: round(float(class_probs[i]), 4)
                            for i in range(4)
                        },
                        "model_version": self.model_version,
                    }
                }
                building_features.append(feature)
                building_id_counter += 1

        return building_features

