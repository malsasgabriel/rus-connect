package main

import (
	"log"
	"math"
	"sort"
	"time"
)

// OrderBookLevel represents a single level in the order book
type OrderBookLevel struct {
	Price    float64 `json:"price"`
	Quantity float64 `json:"quantity"`
	Orders   int     `json:"orders"`
}

// OrderBook represents the complete order book data
type OrderBook struct {
	Symbol    string           `json:"symbol"`
	Timestamp int64            `json:"timestamp"`
	Bids      []OrderBookLevel `json:"bids"` // Buy orders (highest to lowest)
	Asks      []OrderBookLevel `json:"asks"` // Sell orders (lowest to highest)
	LastPrice float64          `json:"last_price"`
}

// OrderBookAnalysis contains analyzed order book metrics
type OrderBookAnalysis struct {
	Symbol        string  `json:"symbol"`
	Timestamp     int64   `json:"timestamp"`
	BidAskSpread  float64 `json:"bid_ask_spread"`
	SpreadPercent float64 `json:"spread_percent"`
	MidPrice      float64 `json:"mid_price"`

	// Market Depth Analysis
	TotalBidVolume   float64 `json:"total_bid_volume"`
	TotalAskVolume   float64 `json:"total_ask_volume"`
	VolumeImbalance  float64 `json:"volume_imbalance"`   // (bids-asks)/(bids+asks)
	MarketDepthRatio float64 `json:"market_depth_ratio"` // Total volume at top levels

	// Order Book Imbalance
	Level1Imbalance   float64 `json:"level1_imbalance"`   // Top level imbalance
	Level5Imbalance   float64 `json:"level5_imbalance"`   // Top 5 levels imbalance
	WeightedImbalance float64 `json:"weighted_imbalance"` // Price-weighted imbalance

	// Large Order Detection
	LargeBidOrders []LargeOrder `json:"large_bid_orders"`
	LargeAskOrders []LargeOrder `json:"large_ask_orders"`
	WhalePresence  float64      `json:"whale_presence"` // Whale activity score

	// Liquidity Metrics
	LiquidityScore  float64 `json:"liquidity_score"`
	EffectiveSpread float64 `json:"effective_spread"`
	MarketImpact    float64 `json:"market_impact"` // Est. impact of large order

	// Market Microstructure
	OrderIntensity     float64 `json:"order_intensity"`     // Orders per price level
	PriceConcentration float64 `json:"price_concentration"` // How concentrated orders are
	DepthResiliency    float64 `json:"depth_resiliency"`    // Order book resilience

	// ML Features
	MLFeatures []float64 `json:"ml_features"` // ML-ready features
}

// LargeOrder represents a significant order in the book
type LargeOrder struct {
	Price        float64 `json:"price"`
	Quantity     float64 `json:"quantity"`
	Value        float64 `json:"value"`        // Price * Quantity
	Distance     float64 `json:"distance"`     // Distance from mid price (%)
	Side         string  `json:"side"`         // "BID" or "ASK"
	Significance float64 `json:"significance"` // How significant this order is
}

// OrderBookEngine analyzes order book data and market microstructure
type OrderBookEngine struct {
	history    map[string][]OrderBookAnalysis // Historical analysis
	maxHistory int                            // Maximum history to keep
	thresholds *AnalysisThresholds            // Analysis thresholds
	lastUpdate map[string]time.Time           // Last update time per symbol
}

// AnalysisThresholds contains configurable thresholds for analysis
type AnalysisThresholds struct {
	LargeOrderThreshold float64 // Minimum size for large order detection
	WhaleThreshold      float64 // Minimum size for whale detection
	ImbalanceThreshold  float64 // Significant imbalance threshold
	SpreadThreshold     float64 // Significant spread threshold
	VolumeThreshold     float64 // Minimum volume for analysis
	DepthLevels         int     // Number of levels to analyze
}

// NewOrderBookEngine creates a new order book analysis engine
func NewOrderBookEngine() *OrderBookEngine {
	return &OrderBookEngine{
		history:    make(map[string][]OrderBookAnalysis),
		maxHistory: 100, // Keep last 100 analyses per symbol
		thresholds: &AnalysisThresholds{
			LargeOrderThreshold: 10000,  // $10,000 USD value
			WhaleThreshold:      100000, // $100,000 USD value
			ImbalanceThreshold:  0.3,    // 30% imbalance
			SpreadThreshold:     0.005,  // 0.5% spread
			VolumeThreshold:     1000,   // Minimum volume
			DepthLevels:         10,     // Analyze top 10 levels
		},
		lastUpdate: make(map[string]time.Time),
	}
}

// AnalyzeOrderBook performs comprehensive order book analysis
func (obe *OrderBookEngine) AnalyzeOrderBook(orderBook *OrderBook) *OrderBookAnalysis {
	if orderBook == nil || len(orderBook.Bids) == 0 || len(orderBook.Asks) == 0 {
		return nil
	}

	analysis := &OrderBookAnalysis{
		Symbol:    orderBook.Symbol,
		Timestamp: orderBook.Timestamp,
	}

	// Basic spread and mid-price analysis
	obe.calculateBasicMetrics(orderBook, analysis)

	// Market depth analysis
	obe.calculateMarketDepth(orderBook, analysis)

	// Order book imbalance analysis
	obe.calculateImbalance(orderBook, analysis)

	// Large order and whale detection
	obe.detectLargeOrders(orderBook, analysis)

	// Liquidity analysis
	obe.calculateLiquidityMetrics(orderBook, analysis)

	// Market microstructure analysis
	obe.calculateMicrostructure(orderBook, analysis)

	// Generate ML features
	obe.generateMLFeatures(orderBook, analysis)

	// Store in history
	obe.storeAnalysis(analysis)

	// Log analysis summary
	log.Printf("📊 Order Book Analysis for %s: Spread=%.4f%%, Imbalance=%.2f, Liquidity=%.2f",
		analysis.Symbol, analysis.SpreadPercent*100, analysis.VolumeImbalance, analysis.LiquidityScore)

	return analysis
}

// calculateBasicMetrics calculates basic order book metrics
func (obe *OrderBookEngine) calculateBasicMetrics(orderBook *OrderBook, analysis *OrderBookAnalysis) {
	if len(orderBook.Bids) == 0 || len(orderBook.Asks) == 0 {
		return
	}

	bestBid := orderBook.Bids[0].Price
	bestAsk := orderBook.Asks[0].Price

	analysis.BidAskSpread = bestAsk - bestBid
	analysis.MidPrice = (bestBid + bestAsk) / 2.0

	if analysis.MidPrice > 0 {
		analysis.SpreadPercent = analysis.BidAskSpread / analysis.MidPrice
	}
}

// calculateMarketDepth calculates market depth metrics
func (obe *OrderBookEngine) calculateMarketDepth(orderBook *OrderBook, analysis *OrderBookAnalysis) {
	bidVolume := 0.0
	askVolume := 0.0

	// Calculate total volume at specified depth levels
	maxLevels := obe.thresholds.DepthLevels

	for i, bid := range orderBook.Bids {
		if i >= maxLevels {
			break
		}
		bidVolume += bid.Quantity
	}

	for i, ask := range orderBook.Asks {
		if i >= maxLevels {
			break
		}
		askVolume += ask.Quantity
	}

	analysis.TotalBidVolume = bidVolume
	analysis.TotalAskVolume = askVolume

	totalVolume := bidVolume + askVolume
	if totalVolume > 0 {
		analysis.VolumeImbalance = (bidVolume - askVolume) / totalVolume
		analysis.MarketDepthRatio = totalVolume / math.Max(float64(len(orderBook.Bids)+len(orderBook.Asks)), 1.0)
	}
}

// calculateImbalance calculates various imbalance metrics
func (obe *OrderBookEngine) calculateImbalance(orderBook *OrderBook, analysis *OrderBookAnalysis) {
	if len(orderBook.Bids) == 0 || len(orderBook.Asks) == 0 {
		return
	}

	// Level 1 imbalance (top of book)
	level1BidVol := orderBook.Bids[0].Quantity
	level1AskVol := orderBook.Asks[0].Quantity
	level1Total := level1BidVol + level1AskVol

	if level1Total > 0 {
		analysis.Level1Imbalance = (level1BidVol - level1AskVol) / level1Total
	}

	// Level 5 imbalance
	level5BidVol := 0.0
	level5AskVol := 0.0

	for i := 0; i < 5 && i < len(orderBook.Bids); i++ {
		level5BidVol += orderBook.Bids[i].Quantity
	}
	for i := 0; i < 5 && i < len(orderBook.Asks); i++ {
		level5AskVol += orderBook.Asks[i].Quantity
	}

	level5Total := level5BidVol + level5AskVol
	if level5Total > 0 {
		analysis.Level5Imbalance = (level5BidVol - level5AskVol) / level5Total
	}

	// Weighted imbalance (considers price distance)
	weightedBidVol := 0.0
	weightedAskVol := 0.0

	midPrice := analysis.MidPrice

	for i, bid := range orderBook.Bids {
		if i >= 10 { // Top 10 levels
			break
		}
		weight := 1.0 / (1.0 + math.Abs(bid.Price-midPrice)/midPrice)
		weightedBidVol += bid.Quantity * weight
	}

	for i, ask := range orderBook.Asks {
		if i >= 10 {
			break
		}
		weight := 1.0 / (1.0 + math.Abs(ask.Price-midPrice)/midPrice)
		weightedAskVol += ask.Quantity * weight
	}

	weightedTotal := weightedBidVol + weightedAskVol
	if weightedTotal > 0 {
		analysis.WeightedImbalance = (weightedBidVol - weightedAskVol) / weightedTotal
	}
}

// detectLargeOrders detects and analyzes large orders
func (obe *OrderBookEngine) detectLargeOrders(orderBook *OrderBook, analysis *OrderBookAnalysis) {
	analysis.LargeBidOrders = make([]LargeOrder, 0)
	analysis.LargeAskOrders = make([]LargeOrder, 0)

	whaleScore := 0.0
	midPrice := analysis.MidPrice

	// Analyze bid side
	for _, bid := range orderBook.Bids {
		value := bid.Price * bid.Quantity
		if value >= obe.thresholds.LargeOrderThreshold {
			distance := math.Abs(bid.Price-midPrice) / midPrice
			significance := value / obe.thresholds.LargeOrderThreshold

			largeOrder := LargeOrder{
				Price:        bid.Price,
				Quantity:     bid.Quantity,
				Value:        value,
				Distance:     distance,
				Side:         "BID",
				Significance: significance,
			}

			analysis.LargeBidOrders = append(analysis.LargeBidOrders, largeOrder)

			if value >= obe.thresholds.WhaleThreshold {
				whaleScore += significance * (1.0 / (1.0 + distance)) // Closer to mid = higher score
			}
		}
	}

	// Analyze ask side
	for _, ask := range orderBook.Asks {
		value := ask.Price * ask.Quantity
		if value >= obe.thresholds.LargeOrderThreshold {
			distance := math.Abs(ask.Price-midPrice) / midPrice
			significance := value / obe.thresholds.LargeOrderThreshold

			largeOrder := LargeOrder{
				Price:        ask.Price,
				Quantity:     ask.Quantity,
				Value:        value,
				Distance:     distance,
				Side:         "ASK",
				Significance: significance,
			}

			analysis.LargeAskOrders = append(analysis.LargeAskOrders, largeOrder)

			if value >= obe.thresholds.WhaleThreshold {
				whaleScore += significance * (1.0 / (1.0 + distance))
			}
		}
	}

	analysis.WhalePresence = whaleScore
}

// calculateLiquidityMetrics calculates liquidity-related metrics
func (obe *OrderBookEngine) calculateLiquidityMetrics(orderBook *OrderBook, analysis *OrderBookAnalysis) {
	// Liquidity score based on volume and spread
	totalVolume := analysis.TotalBidVolume + analysis.TotalAskVolume
	spreadPenalty := 1.0 / (1.0 + analysis.SpreadPercent*100) // Lower spread = higher score
	volumeScore := math.Log(1.0 + totalVolume)

	analysis.LiquidityScore = volumeScore * spreadPenalty

	// Effective spread (includes market impact)
	analysis.EffectiveSpread = analysis.BidAskSpread * 1.1 // Add 10% for market impact

	// Market impact estimation (cost of large order)
	targetVolume := 1000.0 // Simulate 1000 unit order
	impact := obe.estimateMarketImpact(orderBook, targetVolume, "BUY")
	analysis.MarketImpact = impact
}

// calculateMicrostructure calculates market microstructure metrics
func (obe *OrderBookEngine) calculateMicrostructure(orderBook *OrderBook, analysis *OrderBookAnalysis) {
	// Order intensity (average orders per level)
	totalOrders := 0
	totalLevels := len(orderBook.Bids) + len(orderBook.Asks)

	for _, bid := range orderBook.Bids {
		totalOrders += bid.Orders
	}
	for _, ask := range orderBook.Asks {
		totalOrders += ask.Orders
	}

	if totalLevels > 0 {
		analysis.OrderIntensity = float64(totalOrders) / float64(totalLevels)
	}

	// Price concentration (variance of price levels)
	prices := make([]float64, 0)
	volumes := make([]float64, 0)

	for _, bid := range orderBook.Bids {
		prices = append(prices, bid.Price)
		volumes = append(volumes, bid.Quantity)
	}
	for _, ask := range orderBook.Asks {
		prices = append(prices, ask.Price)
		volumes = append(volumes, ask.Quantity)
	}

	analysis.PriceConcentration = obe.calculateConcentration(prices, volumes)

	// Depth resiliency (recovery capability)
	analysis.DepthResiliency = obe.calculateResiliency(orderBook)
}

// generateMLFeatures generates ML-ready features from order book analysis
func (obe *OrderBookEngine) generateMLFeatures(_ *OrderBook, analysis *OrderBookAnalysis) {
	features := []float64{
		analysis.SpreadPercent,
		analysis.VolumeImbalance,
		analysis.Level1Imbalance,
		analysis.Level5Imbalance,
		analysis.WeightedImbalance,
		analysis.LiquidityScore,
		analysis.MarketImpact,
		analysis.WhalePresence,
		analysis.OrderIntensity,
		analysis.PriceConcentration,
		analysis.DepthResiliency,
		float64(len(analysis.LargeBidOrders)),
		float64(len(analysis.LargeAskOrders)),
		analysis.MarketDepthRatio,
		math.Log(1.0 + analysis.TotalBidVolume),
		math.Log(1.0 + analysis.TotalAskVolume),
	}

	analysis.MLFeatures = features
}

// Helper methods

func (obe *OrderBookEngine) estimateMarketImpact(orderBook *OrderBook, volume float64, side string) float64 {
	// Simplified market impact calculation
	remainingVolume := volume
	totalCost := 0.0
	levels := orderBook.Asks

	if side == "SELL" {
		levels = orderBook.Bids
		// Sort bids descending for sell orders
		sort.Slice(levels, func(i, j int) bool {
			return levels[i].Price > levels[j].Price
		})
	}

	for _, level := range levels {
		if remainingVolume <= 0 {
			break
		}

		levelVolume := math.Min(remainingVolume, level.Quantity)
		totalCost += levelVolume * level.Price
		remainingVolume -= levelVolume
	}

	if volume > 0 {
		avgPrice := totalCost / volume
		midPrice := (orderBook.Bids[0].Price + orderBook.Asks[0].Price) / 2.0
		return math.Abs(avgPrice-midPrice) / midPrice
	}

	return 0.0
}

func (obe *OrderBookEngine) calculateConcentration(prices, volumes []float64) float64 {
	if len(prices) == 0 || len(volumes) == 0 {
		return 0.0
	}

	// Calculate volume-weighted price variance
	totalVolume := 0.0
	weightedPrice := 0.0

	for i, volume := range volumes {
		totalVolume += volume
		weightedPrice += prices[i] * volume
	}

	if totalVolume == 0 {
		return 0.0
	}

	avgPrice := weightedPrice / totalVolume
	variance := 0.0

	for i, volume := range volumes {
		diff := prices[i] - avgPrice
		variance += volume * diff * diff
	}

	variance /= totalVolume
	return math.Sqrt(variance) / avgPrice // Normalized standard deviation
}

func (obe *OrderBookEngine) calculateResiliency(orderBook *OrderBook) float64 {
	// Simplified resiliency calculation based on order distribution
	if len(orderBook.Bids) < 5 || len(orderBook.Asks) < 5 {
		return 0.0
	}

	// Check how evenly distributed the volumes are
	bidVolumes := make([]float64, 5)
	askVolumes := make([]float64, 5)

	for i := 0; i < 5; i++ {
		bidVolumes[i] = orderBook.Bids[i].Quantity
		askVolumes[i] = orderBook.Asks[i].Quantity
	}

	bidCV := obe.calculateCV(bidVolumes)
	askCV := obe.calculateCV(askVolumes)

	// Lower coefficient of variation = higher resiliency
	return 1.0 / (1.0 + (bidCV+askCV)/2.0)
}

func (obe *OrderBookEngine) calculateCV(values []float64) float64 {
	if len(values) == 0 {
		return 0.0
	}

	mean := 0.0
	for _, v := range values {
		mean += v
	}
	mean /= float64(len(values))

	if mean == 0 {
		return 0.0
	}

	variance := 0.0
	for _, v := range values {
		diff := v - mean
		variance += diff * diff
	}
	variance /= float64(len(values))

	return math.Sqrt(variance) / mean
}

func (obe *OrderBookEngine) storeAnalysis(analysis *OrderBookAnalysis) {
	symbol := analysis.Symbol

	if obe.history[symbol] == nil {
		obe.history[symbol] = make([]OrderBookAnalysis, 0, obe.maxHistory)
	}

	obe.history[symbol] = append(obe.history[symbol], *analysis)

	// Keep only maxHistory entries
	if len(obe.history[symbol]) > obe.maxHistory {
		obe.history[symbol] = obe.history[symbol][1:]
	}

	obe.lastUpdate[symbol] = time.Now()
}

// GetHistoricalAnalysis returns historical order book analysis
func (obe *OrderBookEngine) GetHistoricalAnalysis(symbol string, limit int) []OrderBookAnalysis {
	if history, exists := obe.history[symbol]; exists {
		if limit > 0 && limit < len(history) {
			return history[len(history)-limit:]
		}
		return history
	}
	return []OrderBookAnalysis{}
}

// DetectMarketRegime detects current market regime based on order book
func (obe *OrderBookEngine) DetectMarketRegime(symbol string) string {
	analyses := obe.GetHistoricalAnalysis(symbol, 10)
	if len(analyses) < 5 {
		return "UNKNOWN"
	}

	recent := analyses[len(analyses)-5:]
	avgImbalance := 0.0
	avgSpread := 0.0
	avgWhalePresence := 0.0

	for _, analysis := range recent {
		avgImbalance += math.Abs(analysis.VolumeImbalance)
		avgSpread += analysis.SpreadPercent
		avgWhalePresence += analysis.WhalePresence
	}

	avgImbalance /= float64(len(recent))
	avgSpread /= float64(len(recent))
	avgWhalePresence /= float64(len(recent))

	// Classify market regime
	if avgSpread > 0.01 && avgImbalance > 0.3 {
		return "VOLATILE"
	} else if avgWhalePresence > 2.0 {
		return "WHALE_DOMINATED"
	} else if avgImbalance < 0.1 && avgSpread < 0.002 {
		return "STABLE"
	} else {
		return "NORMAL"
	}
}
