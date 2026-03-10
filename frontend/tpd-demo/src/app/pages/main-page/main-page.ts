import { Component, computed, signal } from '@angular/core';
import { Navbar } from '../../components/navbar/navbar';
import { ImportImageButton } from '../../components/import-image-button/import-image-button';
import { ModelPredictions } from '../../components/model-predictions/model-predictions';
import { Footer } from '../../components/footer/footer';
import { PredictionService } from '../../services/prediction/prediction.service';
import { Predictions } from '../../models/predictions.model';
import { take } from 'rxjs';
import { NgIf } from '@angular/common';

@Component({
  selector: 'app-main-page',
  imports: [NgIf, Navbar, ImportImageButton, ModelPredictions, Footer],
  templateUrl: './main-page.html',
  styleUrl: './main-page.scss',
})
export class MainPage {
  originalImageUrl = signal('');
  private objectImageUrl: string | null = null;
  predictions = signal<Predictions | null>(null);
  isLoading = signal(false);
  errorMessage = signal<string | null>(null);
  hasUploadedImage = computed(() => !!this.originalImageUrl());

  constructor(private readonly predictionService: PredictionService) {}

  onImageSelected(file: File): void {
    this.errorMessage.set(null);
    this.predictions.set(null);
    this.isLoading.set(true);

    if (this.objectImageUrl) {
      URL.revokeObjectURL(this.objectImageUrl);
    }
    this.objectImageUrl = URL.createObjectURL(file);
    this.originalImageUrl.set(this.objectImageUrl);

    this.predictionService
      .predict(file)
      .pipe(take(1))
      .subscribe({
        next: (predictions) => {
          this.predictions.set(predictions);
          this.isLoading.set(false);
        },
        error: () => {
          this.errorMessage.set(
            'Prediction request failed. Please try another image.',
          );
          this.isLoading.set(false);
        },
      });
  }

  resetPrediction(): void {
    if (this.objectImageUrl) {
      URL.revokeObjectURL(this.objectImageUrl);
      this.objectImageUrl = null;
    }

    this.originalImageUrl.set('');
    this.predictions.set(null);
    this.errorMessage.set(null);
    this.isLoading.set(false);
  }
}
