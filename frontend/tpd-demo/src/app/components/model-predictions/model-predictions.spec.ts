import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ModelPredictions } from './model-predictions';

describe('ModelPredictions', () => {
  let component: ModelPredictions;
  let fixture: ComponentFixture<ModelPredictions>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [ModelPredictions],
    }).compileComponents();

    fixture = TestBed.createComponent(ModelPredictions);
    component = fixture.componentInstance;
    await fixture.whenStable();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
