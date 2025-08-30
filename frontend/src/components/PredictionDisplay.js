import React from 'react';

const PredictionDisplay = ({ prediction, error }) => (
  <>
    {prediction && (
      <div className="prediction-result">
        <strong>Prediction:</strong> {prediction.class}  
        <br />
        <strong>Confidence:</strong> {(prediction.confidence * 100).toFixed(1)}%
      </div>
    )}
    {error && <div className="error-message">{error}</div>}
  </>
);

export default PredictionDisplay;