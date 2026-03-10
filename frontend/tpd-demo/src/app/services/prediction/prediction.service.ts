import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { map, Observable } from 'rxjs';
import { PredictionResponse, Predictions } from '../../models/predictions.model';

@Injectable({
  providedIn: 'root',
})
export class PredictionService {
  private readonly endpoint = '/predict';

  constructor(private readonly http: HttpClient) {}

  predict(imageFile: File): Observable<Predictions> {
    return this.http
      .post<PredictionResponse | { predictions: PredictionResponse }>(
        this.endpoint,
        imageFile,
      )
      .pipe(
        map((response) => {
          const payload =
            'predictions' in response ? response.predictions : response;

          return {
            bbox_x: Number(payload.bbox_x ?? 0),
            bbox_y: Number(payload.bbox_y ?? 0),
            bbox_width: Number(payload.bbox_width ?? 0),
            bbox_height: Number(payload.bbox_height ?? 0),
            cropped_bbox_image_url: this.toImageUrl(
              payload.cropped_bbox_image_url ?? payload.cropped_bbox_image,
            ),

            keypoint_positions: payload.keypoint_positions ?? [],
            keypoint_aggregated_heatmap_url: this.toImageUrl(
              payload.keypoint_aggregated_heatmap_url ??
                payload.keypoint_aggregated_heatmap,
            ),
            keypoint_cropped_image_url: this.toImageUrl(
              payload.keypoint_cropped_image_url ?? payload.keypoint_cropped_image,
            ),

            backhand_conf: Number(payload.backhand_conf ?? 0),
            forehand_conf: Number(payload.forehand_conf ?? 0),
            ready_position_conf: Number(payload.ready_position_conf ?? 0),
            serve_conf: Number(payload.serve_conf ?? 0),
            class_prediction_bar_graph_url: this.toImageUrl(
              payload.class_prediction_bar_graph_url ??
                payload.class_prediction_bar_graph,
            ),

            model_prediction_overlay_image_url: this.toImageUrl(
              payload.model_prediction_overlay_image_url ??
                payload.model_prediction_overlay_image ??
                payload.overlayed_image_url ??
                payload.overlayed_image,
            ),
          };
        }),
      );
  }

  private toImageUrl(rawImage?: string): string {
    if (!rawImage) {
      return '';
    }

    if (rawImage.startsWith('data:image')) {
      return rawImage;
    }

    return `data:image/png;base64,${rawImage}`;
  }
}
