package main

import (
	"log"
	"sync"
	"time"
)

// RateLimiter implements a simple token bucket rate limiter.
type RateLimiter struct {
	capacity     int
	tokens       int
	refillPerSec int
	lastRefill   time.Time
	mu           sync.Mutex
}

func NewRateLimiter(capacity int, refillPerSec int) *RateLimiter {
	rl := &RateLimiter{capacity: capacity, tokens: capacity, refillPerSec: refillPerSec, lastRefill: time.Now()}
	return rl
}

func (r *RateLimiter) refill() {
	now := time.Now()
	elapsed := now.Sub(r.lastRefill).Seconds()
	if elapsed <= 0 {
		return
	}
	added := int(elapsed * float64(r.refillPerSec))
	if added > 0 {
		r.tokens += added
		if r.tokens > r.capacity {
			r.tokens = r.capacity
		}
		r.lastRefill = now
	}
}

// Acquire blocks until it can consume one token.
func (r *RateLimiter) Acquire() {
	for {
		r.mu.Lock()
		r.refill()
		if r.tokens > 0 {
			r.tokens--
			r.mu.Unlock()
			return
		}
		r.mu.Unlock()
		time.Sleep(5 * time.Millisecond) // Wait a short period before retrying
	}
}

// CircuitBreaker is a minimal circuit breaker:
// - closed: allow requests
// - open: block requests
// - half-open: allow a trial request periodically (simplified)
type CircuitBreaker struct {
	failureCount int
	maxFailures  int
	resetAfter   time.Time
	mu           sync.Mutex
	// State: 0=Closed, 1=Open, 2=Half-Open (simplified)
	state int
}

func NewCircuitBreaker(maxFailures int, resetTimeout time.Duration) *CircuitBreaker {
	return &CircuitBreaker{
		maxFailures: maxFailures,
		resetAfter:  time.Now().Add(resetTimeout),
		state:       0, // Initially closed
	}
}

func (cb *CircuitBreaker) IsOpen() bool {
	cb.mu.Lock()
	defer cb.mu.Unlock()

	switch cb.state {
	case 0: // Closed
		return false
	case 1: // Open
		if time.Now().After(cb.resetAfter) {
			cb.state = 2 // Transition to Half-Open
			log.Println("Circuit Breaker: Half-Open state (allowing one trial request)")
			return false // Allow one trial request
		}
		return true // Still Open, block requests
	case 2: // Half-Open
		return false // Allow trial request to test if service recovered
	default:
		return false
	}
}

func (cb *CircuitBreaker) Fail() {
	cb.mu.Lock()
	defer cb.mu.Unlock()

	cb.failureCount++
	if cb.failureCount >= cb.maxFailures && cb.state == 0 {
		cb.state = 1                                     // Transition to Open
		cb.resetAfter = time.Now().Add(10 * time.Second) // Reset after 10 seconds (example)
		log.Printf("Circuit Breaker: Open state (failures: %d)", cb.failureCount)
	}
}

func (cb *CircuitBreaker) Success() {
	cb.mu.Lock()
	defer cb.mu.Unlock()

	switch cb.state {
	case 2: // Half-Open and successful: close it
		cb.state = 0 // Transition to Closed
		cb.failureCount = 0
		log.Println("Circuit Breaker: Closed state (success in Half-Open)")
	case 0: // Closed and successful: reset failures
		cb.failureCount = 0
	}
}
