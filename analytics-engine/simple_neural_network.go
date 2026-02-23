package main

import (
	"log"
	"math"
	"math/rand"
	"time"
)

// MIN_TRAINING_EXAMPLES is the minimum number of examples required to train the neural network.
const MIN_TRAINING_EXAMPLES = 5

// TrainingReport captures a completed training run.
type TrainingReport struct {
	SampleCount             int
	TrainSamples            int
	ValSamples              int
	ClassDown               int
	ClassSideways           int
	ClassUp                 int
	BestValidationAccuracy  float64
}

// NewSimpleNeuralNetwork creates a new simple neural network with hidden layer.
func NewSimpleNeuralNetwork(symbol string) *SimpleNeuralNetwork {
	// 16 input features, 16 hidden units, 3 output classes (DOWN, SIDEWAYS, UP).
	inputSize := 16
	hiddenSize := 16
	outputSize := 3

	// Initialize weights with small random values.
	inputToHiddenWeights := make([][]float64, inputSize)
	for i := range inputToHiddenWeights {
		inputToHiddenWeights[i] = make([]float64, hiddenSize)
		for j := range inputToHiddenWeights[i] {
			inputToHiddenWeights[i][j] = (rand.Float64() - 0.5) * 0.1
		}
	}

	hiddenToOutputWeights := make([][]float64, hiddenSize)
	for i := range hiddenToOutputWeights {
		hiddenToOutputWeights[i] = make([]float64, outputSize)
		for j := range hiddenToOutputWeights[i] {
			hiddenToOutputWeights[i][j] = (rand.Float64() - 0.5) * 0.1
		}
	}

	hiddenBiases := make([]float64, hiddenSize)
	for i := range hiddenBiases {
		hiddenBiases[i] = (rand.Float64() - 0.5) * 0.1
	}

	outputBiases := make([]float64, outputSize)
	for i := range outputBiases {
		outputBiases[i] = (rand.Float64() - 0.5) * 0.1
	}

	return &SimpleNeuralNetwork{
		Symbol:                symbol,
		InputToHiddenWeights:  inputToHiddenWeights,
		HiddenToOutputWeights: hiddenToOutputWeights,
		HiddenBiases:          hiddenBiases,
		OutputBiases:          outputBiases,
		LearningRate:          0.01,
		LastAccuracy:          0.5,
		PredictionCount:       0,
		CorrectCount:          0,
		LastUpdate:            time.Now(),
		Trained:               false,
	}
}

// Predict makes a prediction using the neural network.
func (nn *SimpleNeuralNetwork) Predict(features []float64) (int, float64) {
	classIdx, confidence, _ := nn.PredictWithDistribution(features)
	return classIdx, confidence
}

// PredictWithDistribution returns class, confidence, and class probability distribution.
func (nn *SimpleNeuralNetwork) PredictWithDistribution(features []float64) (int, float64, ClassProbs) {
	if len(features) != len(nn.InputToHiddenWeights) {
		log.Printf("Feature count mismatch: expected %d, got %d", len(nn.InputToHiddenWeights), len(features))
		return 1, 0.5, ClassProbs{Down: 0.25, Sideways: 0.50, Up: 0.25}
	}

	classIdx, probs := nn.predictWithProbs(features)
	confidence := probs[classIdx]
	if confidence < 0 {
		confidence = 0
	}
	if confidence > 1 {
		confidence = 1
	}
	return classIdx, confidence, ClassProbs{
		Down:     probs[0],
		Sideways: probs[1],
		Up:       probs[2],
	}
}

// ApplyTrustStageDampening adjusts confidence based on model trust stage.
// Cold start models get heavily dampened confidence to reflect uncertainty.
// Warming models get slight dampening. Trained models keep raw confidence.
func ApplyTrustStageDampening(rawConfidence float64, trustStage string, predictionCount int) float64 {
	dampenedConfidence := rawConfidence

	switch trustStage {
	case "cold_start":
		// Cold start: heavily dampen confidence to reflect high uncertainty
		// Scale from 0.35 to 0.55 based on prediction count (more data = more trust)
		maxConf := 0.35 + (float64(predictionCount) * 0.004) // 0.35 -> 0.55 over 50 predictions
		if maxConf > 0.55 {
			maxConf = 0.55
		}
		if dampenedConfidence > maxConf {
			dampenedConfidence = maxConf
		}
		// Also apply a floor based on prediction count
		minConf := 0.25 + (float64(predictionCount) * 0.003) // 0.25 -> 0.40 over 50 predictions
		if minConf > 0.40 {
			minConf = 0.40
		}
		if dampenedConfidence < minConf {
			dampenedConfidence = minConf
		}
		// Log for debugging
		if predictionCount < 10 {
			log.Printf("🔍 Cold start dampening: raw=%.4f -> dampened=%.4f (max=%.4f, min=%.4f, count=%d)",
				rawConfidence, dampenedConfidence, maxConf, minConf, predictionCount)
		}

	case "warming":
		// Warming: dampen high confidence slightly, allow moderate confidence
		// Models with 100-500 predictions are still learning
		progress := float64(predictionCount-100) / 400.0 // 0 to 1 over 100-500 predictions
		if progress > 1.0 {
			progress = 1.0
		}
		if progress < 0.0 {
			progress = 0.0
		}
		// Interpolate between dampened (0.65 max) and full confidence
		maxConf := 0.65 + (progress * 0.30) // 0.65 -> 0.95 as model matures
		if dampenedConfidence > maxConf {
			dampenedConfidence = maxConf
		}

	case "trained":
		// Trained: keep raw confidence but cap at 0.95 to avoid overconfidence
		if dampenedConfidence > 0.95 {
			dampenedConfidence = 0.95
		}
	}

	// Ensure confidence stays in valid range
	if dampenedConfidence < 0.0 {
		dampenedConfidence = 0.0
	}
	if dampenedConfidence > 1.0 {
		dampenedConfidence = 1.0
	}

	return dampenedConfidence
}

// Train trains the neural network with time-ordered split and class-weighted loss.
func (nn *SimpleNeuralNetwork) Train(examples []TrainingExample, epochs int, classWeights [3]float64) TrainingReport {
	report := TrainingReport{SampleCount: len(examples)}
	if len(examples) < MIN_TRAINING_EXAMPLES {
		log.Printf("⚠️ [%s] Not enough training examples: %d (need at least %d)", nn.Symbol, len(examples), MIN_TRAINING_EXAMPLES)
		return report
	}

	for _, ex := range examples {
		switch ex.Target {
		case 0:
			report.ClassDown++
		case 1:
			report.ClassSideways++
		case 2:
			report.ClassUp++
		}
	}

	// Time-causal split: first 80% train, latest 20% validation.
	splitIdx := int(0.8 * float64(len(examples)))
	if splitIdx < MIN_TRAINING_EXAMPLES-1 {
		splitIdx = MIN_TRAINING_EXAMPLES - 1
	}
	if splitIdx >= len(examples) {
		splitIdx = len(examples) - 1
	}
	trainExamples := examples[:splitIdx]
	validationExamples := examples[splitIdx:]
	report.TrainSamples = len(trainExamples)
	report.ValSamples = len(validationExamples)

	log.Printf("🚀 Training neural network for %s with %d examples (%d training, %d validation)",
		nn.Symbol, len(examples), len(trainExamples), len(validationExamples))

	bestValidationAccuracy := 0.0
	patience := 0
	maxPatience := 5

	for epoch := 0; epoch < epochs; epoch++ {
		correct := 0
		totalLoss := 0.0

		for _, example := range trainExamples {
			prediction, probs := nn.predictWithProbs(example.Features)
			weight := classWeights[example.Target]
			if weight <= 0 || math.IsNaN(weight) || math.IsInf(weight, 0) {
				weight = 1.0
			}

			targetProb := probs[example.Target]
			if targetProb < 1e-10 {
				targetProb = 1e-10
			}
			loss := -math.Log(targetProb) * weight
			totalLoss += loss

			nn.backpropagate(example.Features, example.Target, probs, weight)

			if prediction == example.Target {
				correct++
			}
		}

		accuracy := float64(correct) / float64(len(trainExamples))
		avgLoss := totalLoss / float64(len(trainExamples))
		validationAccuracy := nn.calculateValidationAccuracy(validationExamples)

		log.Printf("Epoch %d: Loss=%.4f, Train Accuracy=%.4f, Val Accuracy=%.4f",
			epoch, avgLoss, accuracy, validationAccuracy)

		if validationAccuracy > bestValidationAccuracy {
			bestValidationAccuracy = validationAccuracy
			patience = 0
		} else {
			patience++
			if patience >= maxPatience {
				log.Printf("Early stopping at epoch %d", epoch)
				break
			}
		}
	}

	report.BestValidationAccuracy = bestValidationAccuracy
	nn.LastAccuracy = bestValidationAccuracy
	nn.LastUpdate = time.Now()
	nn.Trained = (bestValidationAccuracy >= 0.55)

	if nn.Trained {
		log.Printf("✅ Training completed for %s: Best Validation Accuracy=%.4f (Model is now active)",
			nn.Symbol, bestValidationAccuracy)
	} else {
		log.Printf("✅ Training completed for %s: Best Validation Accuracy=%.4f (Model remains inactive, accuracy too low)",
			nn.Symbol, bestValidationAccuracy)
	}
	return report
}

// calculateValidationAccuracy calculates accuracy on validation set.
func (nn *SimpleNeuralNetwork) calculateValidationAccuracy(validationExamples []TrainingExample) float64 {
	if len(validationExamples) == 0 {
		return 0
	}
	correct := 0
	for _, example := range validationExamples {
		prediction, _ := nn.Predict(example.Features)
		if prediction == example.Target {
			correct++
		}
	}
	return float64(correct) / float64(len(validationExamples))
}

// predictWithProbs returns both prediction and probabilities for training.
func (nn *SimpleNeuralNetwork) predictWithProbs(features []float64) (int, []float64) {
	if len(features) != len(nn.InputToHiddenWeights) {
		log.Printf("Feature count mismatch: expected %d, got %d", len(nn.InputToHiddenWeights), len(features))
		return 1, []float64{0.25, 0.50, 0.25}
	}

	hiddenInputs := make([]float64, len(nn.HiddenBiases))
	for i := range nn.HiddenBiases {
		hiddenInputs[i] = nn.HiddenBiases[i]
		for j, feature := range features {
			hiddenInputs[i] += feature * nn.InputToHiddenWeights[j][i]
		}
	}

	hiddenOutputs := make([]float64, len(hiddenInputs))
	for i, input := range hiddenInputs {
		if input > 0 {
			hiddenOutputs[i] = input
		} else {
			hiddenOutputs[i] = 0
		}
	}

	outputInputs := make([]float64, len(nn.OutputBiases))
	for i := range nn.OutputBiases {
		outputInputs[i] = nn.OutputBiases[i]
		for j, hiddenOutput := range hiddenOutputs {
			outputInputs[i] += hiddenOutput * nn.HiddenToOutputWeights[j][i]
		}
	}

	probs := softmax(outputInputs)
	maxIdx := 0
	maxProb := probs[0]
	for i, prob := range probs {
		if prob > maxProb {
			maxProb = prob
			maxIdx = i
		}
	}
	return maxIdx, probs
}

// backpropagate updates weights using backpropagation algorithm.
func (nn *SimpleNeuralNetwork) backpropagate(features []float64, target int, outputProbs []float64, sampleWeight float64) {
	outputErrors := make([]float64, len(outputProbs))
	for i := range outputErrors {
		outputErrors[i] = outputProbs[i]
		if i == target {
			outputErrors[i] -= 1.0
		}
		outputErrors[i] *= sampleWeight
	}

	for i := range nn.HiddenToOutputWeights {
		for j := range nn.HiddenToOutputWeights[i] {
			nn.HiddenToOutputWeights[i][j] -= nn.LearningRate * outputErrors[j] * nn.getHiddenOutputForBackprop(features, i)
		}
	}

	for j := range nn.OutputBiases {
		nn.OutputBiases[j] -= nn.LearningRate * outputErrors[j]
	}

	hiddenErrors := make([]float64, len(nn.HiddenBiases))
	for i := range nn.HiddenBiases {
		for j := range outputErrors {
			hiddenErrors[i] += outputErrors[j] * nn.HiddenToOutputWeights[i][j]
		}
	}

	hiddenInputs := make([]float64, len(nn.HiddenBiases))
	for i := range nn.HiddenBiases {
		hiddenInputs[i] = nn.HiddenBiases[i]
		for j, feature := range features {
			hiddenInputs[i] += feature * nn.InputToHiddenWeights[j][i]
		}
	}

	for i := range hiddenErrors {
		if hiddenInputs[i] <= 0 {
			hiddenErrors[i] = 0
		}
	}

	for i := range nn.InputToHiddenWeights {
		for j := range nn.InputToHiddenWeights[i] {
			nn.InputToHiddenWeights[i][j] -= nn.LearningRate * hiddenErrors[j] * features[i]
		}
	}
	for j := range nn.HiddenBiases {
		nn.HiddenBiases[j] -= nn.LearningRate * hiddenErrors[j]
	}
}

// getHiddenOutputForBackprop calculates the output of hidden neuron j with ReLU activation.
func (nn *SimpleNeuralNetwork) getHiddenOutputForBackprop(features []float64, j int) float64 {
	sum := nn.HiddenBiases[j]
	for i, feature := range features {
		sum += feature * nn.InputToHiddenWeights[i][j]
	}
	if sum > 0 {
		return sum
	}
	return 0
}

// GetAccuracy returns the current accuracy.
func (nn *SimpleNeuralNetwork) GetAccuracy() float64 {
	if nn.PredictionCount == 0 {
		return 0.5
	}
	return float64(nn.CorrectCount) / float64(nn.PredictionCount)
}

// UpdateAccuracy updates the model accuracy.
func (nn *SimpleNeuralNetwork) UpdateAccuracy(isCorrect bool) {
	nn.PredictionCount++
	if isCorrect {
		nn.CorrectCount++
	}
	nn.LastAccuracy = nn.GetAccuracy()
	nn.LastUpdate = time.Now()
}

// ShouldRetrain checks if the model should be retrained.
func (nn *SimpleNeuralNetwork) ShouldRetrain() bool {
	accuracy := nn.GetAccuracy()
	age := time.Since(nn.LastUpdate)
	if nn.PredictionCount < 100 {
		return accuracy < 0.70 || age > 5*time.Minute || !nn.Trained
	}
	return accuracy < 0.65 || age > 30*time.Minute || !nn.Trained
}

// IsShapeValid validates expected network dimensions.
func (nn *SimpleNeuralNetwork) IsShapeValid() bool {
	if len(nn.InputToHiddenWeights) != 16 {
		return false
	}
	for i := range nn.InputToHiddenWeights {
		if len(nn.InputToHiddenWeights[i]) != 16 {
			return false
		}
	}
	if len(nn.HiddenToOutputWeights) != 16 {
		return false
	}
	for i := range nn.HiddenToOutputWeights {
		if len(nn.HiddenToOutputWeights[i]) != 3 {
			return false
		}
	}
	return len(nn.HiddenBiases) == 16 && len(nn.OutputBiases) == 3
}

// softmax applies softmax function to scores.
func softmax(scores []float64) []float64 {
	maxScore := scores[0]
	for _, score := range scores {
		if score > maxScore {
			maxScore = score
		}
	}

	expSum := 0.0
	expScores := make([]float64, len(scores))
	for i, score := range scores {
		expScores[i] = math.Exp(score - maxScore)
		expSum += expScores[i]
	}

	for i := range expScores {
		expScores[i] /= expSum
	}
	return expScores
}
