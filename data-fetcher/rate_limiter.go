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

// Circuit breaker states.
const (
	cbClosed = iota
	cbOpen
	cbHalfOpen
)

// CircuitBreaker is a minimal circuit breaker:
// - closed: allow requests
// - open: block requests
// - half-open: allow a trial request periodically (simplified)
type CircuitBreaker struct {
	failureCount int
	maxFailures  int
	resetTimeout time.Duration
	resetAfter   time.Time
	mu           sync.Mutex
	state        int
}

func NewCircuitBreaker(maxFailures int, resetTimeout time.Duration) *CircuitBreaker {
	return &CircuitBreaker{
		maxFailures:  maxFailures,
		resetTimeout: resetTimeout,
		resetAfter:   time.Now().Add(resetTimeout),
		state:        cbClosed,
	}
}

func (cb *CircuitBreaker) IsOpen() bool {
	cb.mu.Lock()
	defer cb.mu.Unlock()

	switch cb.state {
	case cbClosed:
		return false
	case cbOpen:
		if time.Now().After(cb.resetAfter) {
			cb.state = cbHalfOpen
			log.Println("Circuit Breaker: Half-Open state (allowing one trial request)")
			return false // Allow one trial request
		}
		return true // Still Open, block requests
	case cbHalfOpen:
		return false // Allow trial request to test if service recovered
	default:
		return false
	}
}

func (cb *CircuitBreaker) Fail() {
	cb.mu.Lock()
	defer cb.mu.Unlock()

	cb.failureCount++

	// A failed trial request in half-open must re-open the breaker immediately.
	// Previously this branch was gated on state == closed, so once the breaker
	// reached half-open it stayed there forever and every request was let
	// through to a failing upstream.
	if cb.state == cbHalfOpen || (cb.state == cbClosed && cb.failureCount >= cb.maxFailures) {
		cb.state = cbOpen
		cb.failureCount = 0
		cb.resetAfter = time.Now().Add(cb.resetTimeout)
		log.Printf("Circuit Breaker: Open state (reset after %v)", cb.resetTimeout)
	}
}

func (cb *CircuitBreaker) Success() {
	cb.mu.Lock()
	defer cb.mu.Unlock()

	switch cb.state {
	case cbHalfOpen: // Half-Open and successful: close it
		cb.state = cbClosed
		cb.failureCount = 0
		log.Println("Circuit Breaker: Closed state (success in Half-Open)")
	case cbClosed: // Closed and successful: reset failures
		cb.failureCount = 0
	}
}
