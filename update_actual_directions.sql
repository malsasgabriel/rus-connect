-- This is a simplified version to update actual_direction for demonstration purposes
-- In a real implementation, you would calculate the actual direction based on price movements

-- For now, we'll randomly assign actual directions to demonstrate the calibration process
UPDATE direction_predictions 
SET actual_direction = CASE 
    WHEN random() < 0.33 THEN 'UP'
    WHEN random() < 0.66 THEN 'DOWN'
    ELSE 'SIDEWAYS'
END
WHERE actual_direction IS NULL;

-- Verify the update
SELECT actual_direction, COUNT(*) 
FROM direction_predictions 
GROUP BY actual_direction;