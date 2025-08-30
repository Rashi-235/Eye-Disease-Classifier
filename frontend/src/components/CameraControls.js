import React from 'react';

const CameraControls = ({
  cameraActive,
  toggleCamera,
  captureImage,
  handleFileChange,
  selectedFile,
  loading,
  handleSubmit
}) => (
  <div className="camera-controls">
    <div className="input-group">
      <label htmlFor="file-upload" className="custom-file-upload">
        Choose Image
      </label>
      <input 
        id="file-upload" 
        type="file" 
        accept="image/*" 
        onChange={handleFileChange} 
      />
      <button 
        type="button" 
        className="camera-btn"
        onClick={toggleCamera}
      >
        {cameraActive ? 'Stop Camera' : 'Use Camera'}
      </button>
      {cameraActive && (
        <button 
          type="button" 
          className="capture-btn"
          onClick={captureImage}
        >
          Capture
        </button>
      )}
      <button 
        type="button" 
        disabled={!selectedFile || loading}
        className="classify-btn"
        onClick={handleSubmit}
      >
        {loading ? 'Classifying…' : 'Classify'}
      </button>
    </div>
  </div>
);

export default CameraControls;