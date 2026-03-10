import { Component } from '@angular/core';
import { MatToolbarModule } from '@angular/material/toolbar';
import { NgIf } from '@angular/common';
import { Input } from '@angular/core';

@Component({
  selector: 'app-navbar',
  imports: [MatToolbarModule, NgIf],
  templateUrl: './navbar.html',
  styleUrl: './navbar.scss',
})
export class Navbar {
  @Input() showSections = false;
}
