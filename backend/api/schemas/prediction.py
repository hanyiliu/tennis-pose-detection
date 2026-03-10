from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    bbox_x: int
    bbox_y: int
    bbox_width: int
    bbox_height: int
    cropped_bbox_image: str

    keypoint_positions: list[list[float]] = Field(default_factory=list)
    keypoint_aggregated_heatmap_url: str
    keypoint_cropped_image: str

    backhand_conf: float
    forehand_conf: float
    ready_position_conf: float
    serve_conf: float
    class_prediction_bar_graph: str

    overlayed_image: str
