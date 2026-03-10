import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ModelFinalPrediction } from './model-final-prediction';

describe('ModelFinalPrediction', () => {
  let component: ModelFinalPrediction;
  let fixture: ComponentFixture<ModelFinalPrediction>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [ModelFinalPrediction],
    }).compileComponents();

    fixture = TestBed.createComponent(ModelFinalPrediction);
    component = fixture.componentInstance;
    await fixture.whenStable();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
