import { Component } from '@angular/core';
import { EventEmitter, Input, Output } from '@angular/core';
import { NgIf } from '@angular/common';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { BackButton } from '../back-button/back-button';
import { ModelFinalPrediction } from '../model-final-prediction/model-final-prediction';
import {
  ModelPredictionItem,
  PredictionImage,
  PredictionValue,
} from '../model-prediction-item/model-prediction-item';
import { Predictions } from '../../models/predictions.model';

@Component({
  selector: 'app-model-predictions',
  imports: [
    NgIf,
    MatProgressSpinnerModule,
    BackButton,
    ModelFinalPrediction,
    ModelPredictionItem,
  ],
  templateUrl: './model-predictions.html',
  styleUrl: './model-predictions.scss',
})
export class ModelPredictions {
  @Input() originalImageUrl = '';
  @Input() predictions: Predictions | null = null;
  @Input() isLoading = false;
  @Input() errorMessage: string | null = null;

  @Output() makeAnotherPrediction = new EventEmitter<void>();

  get stageOneValues(): PredictionValue[] {
    if (!this.predictions) {
      return [];
    }

    return [
      { label: 'bbox_x', value: this.predictions.bbox_x.toFixed(2) },
      { label: 'bbox_y', value: this.predictions.bbox_y.toFixed(2) },
      { label: 'bbox_width', value: this.predictions.bbox_width.toFixed(2) },
      { label: 'bbox_height', value: this.predictions.bbox_height.toFixed(2) },
    ];
  }

  get stageTwoValues(): PredictionValue[] {
    if (!this.predictions) {
      return [];
    }

    return this.predictions.keypoint_positions.map((position, index) => ({
      label: `Keypoint ${index + 1}`,
      value: `[${position.map((value) => Number(value).toFixed(2)).join(', ')}]`,
    }));
  }

  get stageThreeValues(): PredictionValue[] {
    if (!this.predictions) {
      return [];
    }

    return [
      {
        label: 'backhand_conf',
        value: `${(this.predictions.backhand_conf * 100).toFixed(2)}%`,
      },
      {
        label: 'forehand_conf',
        value: `${(this.predictions.forehand_conf * 100).toFixed(2)}%`,
      },
      {
        label: 'ready_position_conf',
        value: `${(this.predictions.ready_position_conf * 100).toFixed(2)}%`,
      },
      {
        label: 'serve_conf',
        value: `${(this.predictions.serve_conf * 100).toFixed(2)}%`,
      },
    ];
  }

  get stageOneImages(): PredictionImage[] {
    if (!this.predictions) {
      return [];
    }

    return [
      {
        label: 'Cropped Bounding Box',
        src: this.predictions.cropped_bbox_image_url,
      },
    ];
  }

  get stageTwoImages(): PredictionImage[] {
    if (!this.predictions) {
      return [];
    }

    return [
      {
        label: 'Keypoint Aggregated Heatmap',
        src: this.predictions.keypoint_aggregated_heatmap_url,
      },
      {
        label: 'Keypoint Overlayed Crop',
        src: this.predictions.keypoint_cropped_image_url,
      },
    ];
  }

  get stageThreeImages(): PredictionImage[] {
    if (!this.predictions) {
      return [];
    }

    return [
      {
        label: 'Class Confidence Bar Graph',
        src: this.predictions.class_prediction_bar_graph_url,
      },
    ];
  }
}
