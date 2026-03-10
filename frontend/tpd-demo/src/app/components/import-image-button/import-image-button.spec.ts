import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ImportImageButton } from './import-image-button';

describe('ImportImageButton', () => {
  let component: ImportImageButton;
  let fixture: ComponentFixture<ImportImageButton>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [ImportImageButton],
    }).compileComponents();

    fixture = TestBed.createComponent(ImportImageButton);
    component = fixture.componentInstance;
    await fixture.whenStable();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
