import { Component } from '@angular/core';
import { EventEmitter, Output } from '@angular/core';
import { MatButtonModule } from '@angular/material/button';

@Component({
  selector: 'app-back-button',
  imports: [MatButtonModule],
  templateUrl: './back-button.html',
  styleUrl: './back-button.scss',
})
export class BackButton {
  @Output() makeAnotherPrediction = new EventEmitter<void>();
}
