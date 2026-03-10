import { Component } from '@angular/core';
import { Input } from '@angular/core';
import { NgFor } from '@angular/common';
import { trigger, state, style, transition, animate } from '@angular/animations';
import { SectionHeader } from '../section-header/section-header';
import { ImageCard } from '../image-card/image-card';

export interface PredictionValue {
  label: string;
  value: string | number;
}

export interface PredictionImage {
  label: string;
  src: string;
}

@Component({
  selector: 'app-model-prediction-item',
  imports: [NgFor, SectionHeader, ImageCard],
  templateUrl: './model-prediction-item.html',
  styleUrl: './model-prediction-item.scss',
  animations: [
    trigger('expandCollapse', [
      state(
        'collapsed',
        style({
          height: '0px',
          opacity: 0,
          overflow: 'hidden',
          marginTop: '0px',
        }),
      ),
      state(
        'expanded',
        style({
          height: '*',
          opacity: 1,
          overflow: 'hidden',
          marginTop: '16px',
        }),
      ),
      transition('collapsed <=> expanded', [animate('220ms ease-in-out')]),
    ]),
  ],
})
export class ModelPredictionItem {
  @Input() mainTitle = '';
  @Input() subTitle = '';
  @Input() values: PredictionValue[] = [];
  @Input() images: PredictionImage[] = [];
  @Input() valuesColumnCount = 1;

  expanded = true;

  toggleExpanded(): void {
    this.expanded = !this.expanded;
  }
}
