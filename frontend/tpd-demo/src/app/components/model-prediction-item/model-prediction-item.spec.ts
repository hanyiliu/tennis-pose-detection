import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ModelPredictionItem } from './model-prediction-item';

describe('ModelPredictionItem', () => {
  let component: ModelPredictionItem;
  let fixture: ComponentFixture<ModelPredictionItem>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [ModelPredictionItem],
    }).compileComponents();

    fixture = TestBed.createComponent(ModelPredictionItem);
    component = fixture.componentInstance;
    await fixture.whenStable();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
