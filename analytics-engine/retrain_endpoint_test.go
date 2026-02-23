package main

import (
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
)

func TestModelRetrainEndpoint(t *testing.T) {
	gin.SetMode(gin.TestMode)
	ae := &AnalyticsEngine{
		symbols:             []string{"BTCUSDT"},
		models:              make(map[string]*SimpleNeuralNetwork),
		trainingData:        make(map[string][]TrainingExample),
		lastTrainedAt:       make(map[string]time.Time),
		pendingExamples:     make(map[string][]PendingExample),
		normalizerManager:   NewFeatureNormalizerManager(16),
		maxTrainingExamples: 500,
	}
	r := gin.New()
	r.POST("/api/v1/model/retrain", ae.handleModelRetrain)

	req := httptest.NewRequest(http.MethodPost, "/api/v1/model/retrain?symbol=BTCUSDT", nil)
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", w.Code)
	}
}
