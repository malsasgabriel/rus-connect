package main

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"
)

func TestHealthzEndpoint(t *testing.T) {
	gin.SetMode(gin.TestMode)
	ae := &AnalyticsEngine{}
	r := gin.New()
	r.GET("/healthz", ae.handleHealthz)

	req := httptest.NewRequest(http.MethodGet, "/healthz", nil)
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", w.Code)
	}
}

func TestReadyzEndpointWithoutDependencies(t *testing.T) {
	gin.SetMode(gin.TestMode)
	ae := &AnalyticsEngine{}
	r := gin.New()
	r.GET("/readyz", ae.handleReadyz)

	req := httptest.NewRequest(http.MethodGet, "/readyz", nil)
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)

	if w.Code != http.StatusServiceUnavailable {
		t.Fatalf("expected 503 without deps, got %d", w.Code)
	}
}
