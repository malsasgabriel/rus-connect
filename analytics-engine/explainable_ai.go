// 🔍 EXPLAINABLE AI SYSTEM - SHAP Values & Feature Importance
package main

import (
	"fmt"
	"math"
	"sort"
	"time"
)

// 🧠 Explainable AI Engine
type ExplainableAI struct {
	shapCalculator   *SHAPCalculator
	explanationCache map[string]*PredictionExplanation
	lastUpdate       time.Time
}

// 📊 Prediction Explanation Structure
type PredictionExplanation struct {
	Symbol     string  `json:"symbol"`
	Timestamp  int64   `json:"timestamp"`
	Prediction string  `json:"prediction"`
	Confidence float64 `json:"confidence"`

	// 🎯 TOP-5 Key Factors
	KeyFactors []FactorExplanation `json:"key_factors"`

	// 📊 SHAP Values
	SHAPValues map[string]float64 `json:"shap_values"`

	// 📈 Feature Importance Chart
	FeatureImportanceChart []FeatureChart `json:"feature_importance_chart"`

	// 🎯 Model Breakdown
	ModelContributions map[string]ModelContribution `json:"model_contributions"`

	// 🛡️ Risk Analysis
	RiskFactors []RiskFactor `json:"risk_factors"`

	// 🎯 Actionable Insights
	ActionableInsights []string `json:"actionable_insights"`
	WarningSignals     []string `json:"warning_signals"`
}

// 🎯 Factor Explanation
type FactorExplanation struct {
	Feature     string  `json:"feature"`
	Importance  float64 `json:"importance"`
	Direction   string  `json:"direction"`
	Value       float64 `json:"value"`
	SHAPValue   float64 `json:"shap_value"`
	Impact      string  `json:"impact"`
	Description string  `json:"description"`
	Reasoning   string  `json:"reasoning"`
}

// 📊 Feature Chart
type FeatureChart struct {
	Feature   string  `json:"feature"`
	Weight    float64 `json:"weight"`
	Impact    float64 `json:"impact"`
	Direction string  `json:"direction"`
	Category  string  `json:"category"`
}

// 🤖 Model Contribution
type ModelContribution struct {
	ModelName    string   `json:"model_name"`
	Confidence   float64  `json:"confidence"`
	Prediction   string   `json:"prediction"`
	Weight       float64  `json:"weight"`
	Contribution float64  `json:"contribution"`
	KeyFeatures  []string `json:"key_features"`
}

// ⚠️ Risk Factor
type RiskFactor struct {
	RiskType    string  `json:"risk_type"`
	Severity    string  `json:"severity"`
	Probability float64 `json:"probability"`
	Impact      float64 `json:"impact"`
	Mitigation  string  `json:"mitigation"`
}

// 🔍 SHAP Calculator
type SHAPCalculator struct {
	baselineValues map[string]float64
	featureWeights map[string]float64
}

// 🚀 Initialize Explainable AI
func NewExplainableAI() *ExplainableAI {
	return &ExplainableAI{
		shapCalculator:   NewSHAPCalculator(),
		explanationCache: make(map[string]*PredictionExplanation),
		lastUpdate:       time.Now(),
	}
}

func NewSHAPCalculator() *SHAPCalculator {
	weights := map[string]float64{
		"RSI_14":             0.15,
		"MACD_Histogram":     0.12,
		"Volume":             0.10,
		"BTC_Correlation":    0.08,
		"OrderBookImbalance": 0.07,
		"TimeOfDay":          0.05,
		"FibRetracement":     0.06,
		"TrendStrengthML":    0.08,
		"BB_Position":        0.07,
		"ATR":                0.05,
	}

	return &SHAPCalculator{
		baselineValues: make(map[string]float64),
		featureWeights: weights,
	}
}

// 🧠 MAIN EXPLANATION METHOD
func (xai *ExplainableAI) ExplainPrediction(features *AdvancedFeatures, prediction *ModelPrediction) []*FeatureImportance {
	// 1. Calculate SHAP values
	shapValues := xai.shapCalculator.CalculateSHAPValues(features, prediction)

	// 2. Generate key factors (top 5)
	keyFactors := xai.generateKeyFactors(features, shapValues, prediction)

	// 3. Create feature chart
	featureChart := xai.createFeatureChart(shapValues, features)

	// 4. Analyze model contributions
	modelContributions := xai.analyzeModelContributions(prediction)

	// 5. Identify risks
	riskFactors := xai.identifyRiskFactors(features, prediction)

	// 6. Generate insights
	insights, warnings := xai.generateInsights(features, prediction, keyFactors)

	// Create explanation
	explanation := &PredictionExplanation{
		Symbol:                 features.Symbol,
		Timestamp:              time.Now().Unix(),
		Prediction:             prediction.Prediction,
		Confidence:             prediction.Confidence,
		KeyFactors:             keyFactors,
		SHAPValues:             shapValues,
		FeatureImportanceChart: featureChart,
		ModelContributions:     modelContributions,
		RiskFactors:            riskFactors,
		ActionableInsights:     insights,
		WarningSignals:         warnings,
	}

	// Cache explanation
	xai.explanationCache[features.Symbol] = explanation

	// Convert to FeatureImportance for compatibility
	result := make([]*FeatureImportance, len(keyFactors))
	for i, factor := range keyFactors {
		result[i] = &FeatureImportance{
			FeatureName: factor.Feature,
			Importance:  factor.Importance,
			Impact:      factor.Direction,
			Value:       factor.Value,
			SHAPValue:   factor.SHAPValue,
			Description: factor.Description,
		}
	}

	return result
}

// 🔍 Calculate SHAP Values
func (shap *SHAPCalculator) CalculateSHAPValues(features *AdvancedFeatures, prediction *ModelPrediction) map[string]float64 {
	shapValues := make(map[string]float64)
	baseline := 0.5

	// Key features to analyze
	featureMap := map[string]float64{
		"RSI_14":             features.BasicFeatures.RSI_14 / 100.0,
		"MACD_Histogram":     math.Tanh(features.BasicFeatures.MACD_Histogram),
		"Volume":             math.Min(1.0, features.BasicFeatures.Volume/1000000.0),
		"BTC_Correlation":    features.BTCCorrelation,
		"OrderBookImbalance": features.OrderBookImbalance,
		"TimeOfDay":          features.TimeOfDay,
		"FibRetracement":     features.FibRetracementLevel,
		"TrendStrengthML":    features.TrendStrengthML,
		"BB_Position":        features.BasicFeatures.BB_Position,
		"ATR":                math.Min(1.0, features.BasicFeatures.ATR/100.0),
	}

	totalContribution := 0.0
	for featureName, featureValue := range featureMap {
		weight := shap.featureWeights[featureName]
		direction := shap.getFeatureDirection(featureName, featureValue)

		shapValue := (featureValue - baseline) * weight * direction
		shapValues[featureName] = shapValue
		totalContribution += math.Abs(shapValue)
	}

	// Normalize SHAP values
	if totalContribution > 0 {
		for feature, value := range shapValues {
			shapValues[feature] = value / totalContribution
		}
	}

	return shapValues
}

// 🎯 Generate Key Factors
func (xai *ExplainableAI) generateKeyFactors(features *AdvancedFeatures, shapValues map[string]float64, _ *ModelPrediction) []FactorExplanation {
	type featureShap struct {
		feature string
		shap    float64
		absShap float64
	}

	var featureList []featureShap
	for feature, shap := range shapValues {
		featureList = append(featureList, featureShap{
			feature: feature,
			shap:    shap,
			absShap: math.Abs(shap),
		})
	}

	// Sort by importance
	sort.Slice(featureList, func(i, j int) bool {
		return featureList[i].absShap > featureList[j].absShap
	})

	// Top 5 factors
	keyFactors := make([]FactorExplanation, 0, 5)
	for i := 0; i < 5 && i < len(featureList); i++ {
		feature := featureList[i]
		factor := xai.createFactorExplanation(feature.feature, feature.shap, features)
		keyFactors = append(keyFactors, factor)
	}

	return keyFactors
}

// 🔍 Create Factor Explanation
func (xai *ExplainableAI) createFactorExplanation(featureName string, shapValue float64, features *AdvancedFeatures) FactorExplanation {
	value := xai.getFeatureValue(featureName, features)

	direction := "NEUTRAL"
	if shapValue > 0.05 {
		direction = "BULLISH"
	} else if shapValue < -0.05 {
		direction = "BEARISH"
	}

	impact := "LOW"
	absShap := math.Abs(shapValue)
	if absShap > 0.2 {
		impact = "HIGH"
	} else if absShap > 0.1 {
		impact = "MEDIUM"
	}

	description := xai.generateDescription(featureName, value, direction)
	reasoning := xai.generateReasoning(featureName, shapValue, direction)

	return FactorExplanation{
		Feature:     featureName,
		Importance:  absShap,
		Direction:   direction,
		Value:       value,
		SHAPValue:   shapValue,
		Impact:      impact,
		Description: description,
		Reasoning:   reasoning,
	}
}

// Helper functions
func (shap *SHAPCalculator) getFeatureDirection(feature string, value float64) float64 {
	switch feature {
	case "RSI_14":
		if value > 0.7 {
			return -1.0 // Overbought
		} else if value < 0.3 {
			return 1.0 // Oversold
		}
		return 0.0
	case "Volume":
		return 1.0 // Higher volume = stronger signal
	case "BTC_Correlation":
		return value
	default:
		return math.Tanh(value - 0.5)
	}
}

func (xai *ExplainableAI) getFeatureValue(featureName string, features *AdvancedFeatures) float64 {
	switch featureName {
	case "RSI_14":
		return features.BasicFeatures.RSI_14
	case "MACD_Histogram":
		return features.BasicFeatures.MACD_Histogram
	case "Volume":
		return features.BasicFeatures.Volume
	case "BTC_Correlation":
		return features.BTCCorrelation
	case "OrderBookImbalance":
		return features.OrderBookImbalance
	case "TimeOfDay":
		return features.TimeOfDay
	case "FibRetracement":
		return features.FibRetracementLevel
	case "TrendStrengthML":
		return features.TrendStrengthML
	default:
		return 0.5
	}
}

func (xai *ExplainableAI) generateDescription(featureName string, value float64, direction string) string {
	descriptions := map[string]string{
		"RSI_14":             fmt.Sprintf("RSI %.1f indicates %s momentum", value, direction),
		"MACD_Histogram":     fmt.Sprintf("MACD histogram %.3f shows %s trend", value, direction),
		"Volume":             fmt.Sprintf("Volume %.0f shows %s participation", value, direction),
		"BTC_Correlation":    fmt.Sprintf("BTC correlation %.2f indicates %s dependency", value, direction),
		"OrderBookImbalance": fmt.Sprintf("Order book imbalance %.2f shows %s pressure", value, direction),
	}

	if desc, exists := descriptions[featureName]; exists {
		return desc
	}
	return fmt.Sprintf("%s: %.3f (%s signal)", featureName, value, direction)
}

func (xai *ExplainableAI) generateReasoning(_ string, shapValue float64, direction string) string {
	if math.Abs(shapValue) < 0.05 {
		return "Minimal impact on prediction"
	}

	impact := "supports"
	if shapValue < 0 {
		impact = "opposes"
	}

	return fmt.Sprintf("This %s signal %s the prediction with %.1f%% influence",
		direction, impact, math.Abs(shapValue)*100)
}

func (xai *ExplainableAI) createFeatureChart(shapValues map[string]float64, _ *AdvancedFeatures) []FeatureChart {
	chart := make([]FeatureChart, 0, len(shapValues))

	categories := map[string]string{
		"RSI_14":             "TECHNICAL",
		"Volume":             "VOLUME",
		"BTC_Correlation":    "CORRELATION",
		"OrderBookImbalance": "ORDER_BOOK",
		"TimeOfDay":          "TIME",
	}

	for feature, shap := range shapValues {
		direction := "POSITIVE"
		if shap < 0 {
			direction = "NEGATIVE"
		}

		category := "OTHER"
		if cat, exists := categories[feature]; exists {
			category = cat
		}

		chart = append(chart, FeatureChart{
			Feature:   feature,
			Weight:    math.Abs(shap),
			Impact:    shap,
			Direction: direction,
			Category:  category,
		})
	}

	return chart
}

func (xai *ExplainableAI) analyzeModelContributions(prediction *ModelPrediction) map[string]ModelContribution {
	return map[string]ModelContribution{
		prediction.ModelType: {
			ModelName:    prediction.ModelType,
			Confidence:   prediction.Confidence,
			Prediction:   prediction.Prediction,
			Weight:       1.0,
			Contribution: prediction.Confidence,
			KeyFeatures:  []string{"RSI_14", "Volume", "MACD_Histogram"},
		},
	}
}

func (xai *ExplainableAI) identifyRiskFactors(features *AdvancedFeatures, prediction *ModelPrediction) []RiskFactor {
	risks := []RiskFactor{}

	// High volatility risk
	if features.BasicFeatures.ATR > 0.05 {
		risks = append(risks, RiskFactor{
			RiskType:    "VOLATILITY",
			Severity:    "HIGH",
			Probability: 0.7,
			Impact:      0.4,
			Mitigation:  "Use smaller position size and wider stops",
		})
	}

	// Low confidence risk
	if prediction.Confidence < 0.7 {
		risks = append(risks, RiskFactor{
			RiskType:    "CONFIDENCE",
			Severity:    "MEDIUM",
			Probability: 0.5,
			Impact:      0.3,
			Mitigation:  "Wait for higher confidence signal",
		})
	}

	return risks
}

func (xai *ExplainableAI) generateInsights(features *AdvancedFeatures, prediction *ModelPrediction, factors []FactorExplanation) ([]string, []string) {
	insights := []string{}
	warnings := []string{}

	// Generate insights based on key factors
	for _, factor := range factors {
		if factor.Impact == "HIGH" {
			insight := fmt.Sprintf("Strong %s signal from %s", factor.Direction, factor.Feature)
			insights = append(insights, insight)
		}
	}

	// Generate warnings
	if prediction.Confidence < 0.65 {
		warnings = append(warnings, "Low confidence prediction - consider waiting")
	}

	if features.BasicFeatures.ATR > 0.05 {
		warnings = append(warnings, "High volatility detected - manage risk carefully")
	}

	return insights, warnings
}

// Get cached explanation for frontend
func (xai *ExplainableAI) GetExplanation(symbol string) *PredictionExplanation {
	if explanation, exists := xai.explanationCache[symbol]; exists {
		return explanation
	}
	return nil
}
