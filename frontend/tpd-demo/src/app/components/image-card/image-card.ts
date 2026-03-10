import { Component } from '@angular/core';
import { Input } from '@angular/core';
import { NgIf } from '@angular/common';

@Component({
  selector: 'app-image-card',
  imports: [NgIf],
  templateUrl: './image-card.html',
  styleUrl: './image-card.scss',
})
export class ImageCard {
  @Input() src = '';
  @Input() alt = 'Prediction image';
}
