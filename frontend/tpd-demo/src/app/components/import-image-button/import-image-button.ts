import { Component } from '@angular/core';
import { Output, EventEmitter } from '@angular/core';
import { MatButtonModule } from '@angular/material/button';

@Component({
  selector: 'app-import-image-button',
  imports: [MatButtonModule],
  templateUrl: './import-image-button.html',
  styleUrl: './import-image-button.scss',
})
export class ImportImageButton {
  @Output() imageSelected = new EventEmitter<File>();

  onFileInputChange(event: Event): void {
    const input = event.target as HTMLInputElement;
    const file = input.files?.[0];

    if (!file) {
      return;
    }

    this.imageSelected.emit(file);
    input.value = '';
  }
}
