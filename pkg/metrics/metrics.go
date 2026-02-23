package metrics

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

var (
	PredictionsTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "predictions_total",
			Help: "Total number of predictions made",
		},
		[]string{"symbol", "direction"},
	)

	DatabaseQueriesDuration = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name: "db_queries_duration_seconds",
			Help: "Database query duration",
		},
		[]string{"operation"},
	)
)
