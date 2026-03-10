import { Component } from '@angular/core';
import { Input } from '@angular/core';
import { SectionHeader } from '../section-header/section-header';
import { ImageCard } from '../image-card/image-card';

@Component({
  selector: 'app-model-final-prediction',
  imports: [SectionHeader, ImageCard],
  templateUrl: './model-final-prediction.html',
  styleUrl: './model-final-prediction.scss',
})
export class ModelFinalPrediction {
  @Input() originalImageUrl = '';
  @Input() overlayImageUrl = '';
}
