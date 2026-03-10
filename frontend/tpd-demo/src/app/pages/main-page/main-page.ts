import { Component } from '@angular/core';
import { Navbar } from '../../components/navbar/navbar';
import { ImportImageButton } from '../../components/import-image-button/import-image-button';
import { ModelPredictions } from '../../components/model-predictions/model-predictions';
import { Footer } from '../../components/footer/footer';
import { PredictionService } from '../../services/prediction/prediction.service';
import { Predictions } from '../../models/predictions.model';
import { finalize } from 'rxjs';
import { NgIf } from '@angular/common';

@Component({
  selector: 'app-main-page',
  imports: [NgIf, Navbar, ImportImageButton, ModelPredictions, Footer],
  templateUrl: './main-page.html',
  styleUrl: './main-page.scss',
})
export class MainPage {
  originalImageUrl = '';
  predictions: Predictions | null = null;
  isLoading = false;
  errorMessage: string | null = null;

  constructor(private readonly predictionService: PredictionService) {}

  get hasUploadedImage(): boolean {
    return !!this.originalImageUrl;
  }

  async onImageSelected(file: File): Promise<void> {
    this.errorMessage = null;
    this.predictions = null;
    this.isLoading = true;

    try {
      this.originalImageUrl = await this.fileToDataUrl(file);
    } catch {
      this.errorMessage = 'Could not read the selected image.';
      this.isLoading = false;
      return;
    }

    this.predictionService
      .predict(file)
      .pipe(finalize(() => (this.isLoading = false)))
      .subscribe({
        next: (predictions) => {
          this.predictions = predictions;
        },
        error: () => {
          this.errorMessage = 'Prediction request failed. Please try another image.';
        },
      });
  }

  resetPrediction(): void {
    this.originalImageUrl = '';
    this.predictions = null;
    this.errorMessage = null;
    this.isLoading = false;
  }

  private fileToDataUrl(file: File): Promise<string> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(String(reader.result ?? ''));
      reader.onerror = () => reject(reader.error);
      reader.readAsDataURL(file);
    });
  }
}
