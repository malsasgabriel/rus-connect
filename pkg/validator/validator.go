package validator

import (
	"errors"
	"regexp"
	"time"
)

var symbolRegex = regexp.MustCompile(`^[A-Z]{2,10}USDT$`)

func ValidateSymbol(symbol string) error {
	if !symbolRegex.MatchString(symbol) {
		return errors.New("invalid symbol format")
	}
	return nil
}

func ValidateTimestamp(ts int64) error {
	if ts < 0 || ts > time.Now().Unix()+86400 {
		return errors.New("invalid timestamp")
	}
	return nil
}

func ValidateConfidence(conf float64) error {
	if conf < 0 || conf > 1 {
		return errors.New("confidence must be between 0 and 1")
	}
	return nil
}

func ValidatePrice(price float64) error {
	if price <= 0 {
		return errors.New("price must be positive")
	}
	return nil
}
