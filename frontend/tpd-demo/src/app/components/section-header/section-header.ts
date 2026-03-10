import { Component } from '@angular/core';
import { EventEmitter, Input, Output } from '@angular/core';
import { MatButtonModule } from '@angular/material/button';
import { NgIf } from '@angular/common';

@Component({
  selector: 'app-section-header',
  imports: [MatButtonModule, NgIf],
  templateUrl: './section-header.html',
  styleUrl: './section-header.scss',
})
export class SectionHeader {
  @Input() mainTitle = '';
  @Input() subTitle = '';
  @Input() showToggle = true;
  @Input() expanded = true;

  @Output() toggleSection = new EventEmitter<void>();
}
