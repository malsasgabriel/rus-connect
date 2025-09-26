# 🚀 Advanced ML Cryptocurrency Trading System - 65-75% Accuracy Target

A high-performance, self-learning cryptocurrency trading system featuring **Ensemble AI Models** with LSTM + XGBoost + Transformer + Meta-Learner for achieving **65-75% prediction accuracy** with explainable AI and adaptive risk management.

## 🎯 System Overview

This system transforms raw cryptocurrency market data into high-accuracy trading signals using:
- **🧠 Ensemble AI Models** - LSTM + XGBoost + Transformer + Meta-Learner combination
- **📊 Advanced Feature Engineering** - 50+ normalized features including Fibonacci, Order Book, Time-based
- **🔍 Explainable AI** - SHAP values and detailed prediction explanations
- **🛡️ Adaptive Risk Management** - Dynamic thresholds and position sizing
- **📈 Real-time Learning** - Continuous model improvement from market outcomes
- **🎯 Target Accuracy: 65-75%** - Honest, high-confidence signals only

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Fetcher  │───▶│ Analytics Engine │───▶│   API Gateway   │
│   (Bybit API)   │    │  (Advanced ML)   │    │  (REST + WS)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│     Kafka       │    │   PostgreSQL    │    │    Frontend     │
│  (Message Bus)  │    │   (Storage)     │    │  (React + TS)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📋 Prerequisites

- **Docker Desktop** (v4.0+) - **MUST BE RUNNING**
- **Docker Compose** (v2.0+)
- **PowerShell** (Windows) or **Bash** (Linux/Mac)
- **8GB RAM** minimum (16GB recommended)
- **5GB free disk space**
- **Internet connection** for API access

> ⚠️ **IMPORTANT**: Make sure Docker Desktop is running before executing any commands!

## 🚀 Quick Start

### 1. Clone and Navigate
```powershell
git clone <repository-url>
cd rus-connect
```

### 2. Clone and Navigate
```powershell
git clone <repository-url>
cd rus-connect
```

### 3. Build and Start Complete System
```powershell
docker-compose build
docker-compose up -d
```

### 4. Verify Services
```powershell
docker-compose ps
```

Expected output:
```
NAME                             STATUS
rus-connect-analytics-engine-1   Up
rus-connect-api-gateway-1        Up  
rus-connect-data-fetcher-1       Up
rus-connect-frontend-1           Up
rus-connect-kafka-1              Up (healthy)
rus-connect-postgres-1           Up (healthy)
rus-connect-redis-1              Up (healthy)
```

### 4. Access Web Interface
Open browser: **http://localhost:3000**

## 🔧 Individual Module Management

### Analytics Engine (Core ML)
```powershell
# Start only analytics engine
docker-compose up analytics-engine -d

# View real-time logs
docker-compose logs -f analytics-engine

# Check ML predictions
docker-compose logs analytics-engine | findstr "HONEST SIGNAL"

# Restart with new code
docker-compose build analytics-engine
docker-compose up analytics-engine -d
```

### Data Fetcher (Market Data)
```powershell
# Start data fetcher
docker-compose up data-fetcher -d

# Monitor data flow
docker-compose logs -f data-fetcher

# Check API connections
docker-compose logs data-fetcher | findstr "Connected to Bybit"
```

### Frontend (Web Interface)
```powershell
# Start frontend only
docker-compose up frontend -d

# Rebuild frontend
docker-compose build frontend
docker-compose up frontend -d

# Check frontend logs
docker-compose logs -f frontend
```

### API Gateway (REST API)
```powershell
# Start API gateway
docker-compose up api-gateway -d

# Test REST endpoints
curl http://localhost:8080/api/market-data
curl http://localhost:8080/api/trading-signals

# Monitor API requests
docker-compose logs -f api-gateway
```

### Database (PostgreSQL)
```powershell
# Start database
docker-compose up postgres -d

# Connect to database
docker-compose exec postgres psql -U admin -d predpump

# View ML model data
docker-compose exec postgres psql -U admin -d predpump -c "\dt"
```

### Message Bus (Kafka)
```powershell
# Start Kafka
docker-compose up kafka -d

# List topics
docker-compose exec kafka kafka-topics.sh --list --bootstrap-server localhost:9092

# Monitor messages
docker-compose exec kafka kafka-console-consumer.sh --topic direction_signals --bootstrap-server localhost:9092
```

## 🧪 Testing & Validation

### System Health Check
```powershell
# Check all services
docker-compose ps

# Verify ML signals generation
docker-compose logs analytics-engine | findstr "🎯 HONEST SIGNAL"

# Check signal confidence levels
docker-compose logs analytics-engine | findstr "confidence"
```

### ML Model Testing
```powershell
# Monitor advanced ML engine initialization
docker-compose logs analytics-engine | findstr "🧠 Advanced ML Engine"

# Check model accuracy improvements
docker-compose logs analytics-engine | findstr "accuracy"

# Verify training progress
docker-compose logs analytics-engine | findstr "Training completed"
```

### Database Testing
```powershell
# Connect to database
docker-compose exec postgres psql -U admin -d predpump

# Check market data storage
SELECT COUNT(*) FROM candle_cache;

# Check ML models
SELECT symbol, accuracy FROM smart_models;

# Check trading signals
SELECT symbol, prediction, confidence FROM direction_predictions ORDER BY created_at DESC LIMIT 10;

# Check performance metrics
SELECT * FROM model_performance ORDER BY created_at DESC LIMIT 5;
```

### API Testing
```powershell
# Test market data endpoint
curl http://localhost:8080/api/market-data | jq

# Test trading signals endpoint  
curl http://localhost:8080/api/trading-signals | jq

# Test WebSocket connection
# Use browser dev tools: new WebSocket('ws://localhost:8080/ws')
```

### Frontend Testing
1. Open **http://localhost:3000**
2. Check **Trading Signals Dashboard** table
3. Verify signals appear without popups
4. Monitor real-time updates
5. Check signal confidence levels (should be 65%+)

## 🐛 Troubleshooting

### Common Issues

#### 1. Docker Desktop Not Running (Windows)
**Error**: `unable to get image 'rus-connect-analytics-engine': error during connect: Get "http://%2F%2F.%2Fpipe%2FdockerDesktopLinuxEngine/v1.51/images/..." open //./pipe/dockerDesktopLinuxEngine: The system cannot find the file specified.`

**Solution**:
1. **Start Docker Desktop**:
   - Click Start Menu → Search "Docker Desktop" → Open
   - Wait for Docker Engine to start (green whale icon in system tray)
   - Ensure "Use WSL 2 based engine" is enabled in Settings

2. **Verify Docker is running**:
   ```powershell
   docker --version
   docker ps
   ```

3. **If Docker Desktop won't start**:
   - Run as Administrator
   - Enable "Windows Subsystem for Linux" feature
   - Restart computer if prompted
   - Update Docker Desktop to latest version

#### 2. Services Won't Start
```powershell
# Check Docker daemon
docker --version
docker-compose --version

# Restart Docker Desktop
# Then retry: docker-compose up -d
```

#### 2. Database Connection Failed
```powershell
# Check PostgreSQL logs
docker-compose logs postgres

# Restart database
docker-compose restart postgres

# Wait for health check
docker-compose ps | findstr postgres
```

#### 3. Low Signal Confidence
```powershell
# Check ML engine logs
docker-compose logs analytics-engine | findstr "confidence"

# Verify advanced ML engine is active
docker-compose logs analytics-engine | findstr "🧠 Advanced ML"

# Check historical data loading
docker-compose logs analytics-engine | findstr "historical"
```

#### 4. No Trading Signals
```powershell
# Check data fetcher
docker-compose logs data-fetcher | findstr "Connected"

# Verify Kafka messages
docker-compose exec kafka kafka-topics.sh --list --bootstrap-server localhost:9092

# Check analytics engine processing
docker-compose logs analytics-engine | findstr "candles needed"
```

#### 5. Frontend Not Loading
```powershell
# Check frontend service
docker-compose logs frontend

# Rebuild and restart
docker-compose build frontend
docker-compose up frontend -d

# Check port binding
netstat -an | findstr :3000
```

### Debug Commands

```powershell
# View all service logs
docker-compose logs

# Check specific service health
docker-compose exec analytics-engine ps aux

# Monitor resource usage
docker stats

# Clean restart everything
docker-compose down
docker-compose up -d

# Reset database (WARNING: deletes all data)
docker-compose down -v
docker-compose up -d
```

## 📊 Performance Monitoring

### Key Metrics to Monitor

1. **Signal Quality**
   ```powershell
   # Check confidence levels
   docker-compose logs analytics-engine | findstr "confidence" | tail -20
   ```

2. **Model Accuracy**
   ```powershell
   # Check training progress
   docker-compose logs analytics-engine | findstr "accuracy" | tail -10
   ```

3. **Data Flow**
   ```powershell
   # Monitor candle processing
   docker-compose logs analytics-engine | findstr "candles" | tail -15
   ```

4. **System Performance**
   ```powershell
   # Resource usage
   docker stats --no-stream
   ```

### Expected Performance
- **Signal Confidence:** 65-80% (honest signals only)
- **Prediction Frequency:** 1-5 signals per hour per symbol
- **Model Accuracy:** Improves over time (55%+ baseline)
- **Response Time:** <2 seconds for real-time updates

## 🔧 Configuration

### Environment Variables
Create `.env` file:
```env
# Database
PG_DSN=host=postgres user=admin password=password dbname=predpump sslmode=disable

# Kafka
KAFKA_BROKERS=kafka:9092

# API
API_PORT=8080

# ML Settings
CONFIDENCE_THRESHOLD=0.65
MIN_ACCURACY=0.55
```

### Advanced ML Configuration
Edit `analytics-engine/advanced_ml_engine.go`:
```go
confidenceThreshold: 0.65  // Minimum 65% for signal emission
minAccuracy: 0.55          // Minimum model accuracy 55%
adaptiveLearning: true     // Enable continuous learning
```

## 📚 API Documentation

### REST Endpoints

#### Market Data
```
GET /api/market-data
Response: Current market data for all symbols
```

#### Trading Signals  
```
GET /api/trading-signals
Response: Latest ML trading signals with confidence levels
```

### WebSocket Events
```javascript
const ws = new WebSocket('ws://localhost:8080/ws');
ws.onmessage = (event) => {
    const signal = JSON.parse(event.data);
    console.log('New signal:', signal);
};
```

## 🗄️ Database Schema

### Key Tables
- **`candle_cache`** - Historical market data (OHLCV)
- **`smart_models`** - ML model weights (binary storage)
- **`direction_predictions`** - Trading signals with outcomes
- **`model_performance`** - Accuracy tracking
- **`training_examples`** - ML training data

### Query Examples
```sql
-- Check latest signals
SELECT symbol, prediction, confidence, created_at 
FROM direction_predictions 
ORDER BY created_at DESC LIMIT 10;

-- Model performance
SELECT symbol, accuracy, prediction_count 
FROM smart_models;

-- Market data summary
SELECT symbol, COUNT(*) as candle_count 
FROM candle_cache 
GROUP BY symbol;
```

## 🛡️ Security Notes

- Database credentials are in Docker network only
- No external API keys exposed
- WebSocket connections are local only
- All data stored locally

## 🔄 Updates & Maintenance

### Update System
```powershell
# Pull latest code
git pull origin main

# Rebuild and restart
docker-compose build
docker-compose up -d
```

### Backup Data
```powershell
# Backup database
docker-compose exec postgres pg_dump -U admin predpump > backup.sql

# Backup volume data
docker cp rus-connect-postgres-1:/var/lib/postgresql/data ./postgres-backup
```

### Clean Installation
```powershell
# Remove all containers and volumes
docker-compose down -v

# Remove images  
docker-compose down --rmi all

# Fresh start
docker-compose up -d
```

## 📈 Expected Results

After successful deployment, you should see:
- **🎯 High-confidence signals** (65-80% confidence)
- **🧠 Continuous learning** from market outcomes  
- **📊 Real-time signal table** on frontend
- **🗄️ Persistent storage** of all data
- **⚡ Fast performance** with compact storage

## 🤝 Support

For issues or questions:
1. Check logs: `docker-compose logs [service-name]`
2. Verify configuration files
3. Restart specific services
4. Clean restart if needed

The system is designed to be self-healing and will automatically:
- Reconnect to data sources
- Retrain models on poor performance
- Persist data between restarts
- Generate only honest, high-confidence signals

**Happy Trading! 🚀**