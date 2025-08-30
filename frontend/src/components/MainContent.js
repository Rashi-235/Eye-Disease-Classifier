import React from 'react';
import CameraControls from './CameraControls';
import ImagePreview from './ImagePreview';
import PredictionDisplay from './PredictionDisplay';

const MainContent = ({
  cameraActive,
  videoRef,
  previewUrl,
  toggleCamera,
  captureImage,
  handleFileChange,
  selectedFile,
  loading,
  handleSubmit,
  prediction,
  error
}) => (
  <div className="main-content">
    <CameraControls 
      cameraActive={cameraActive}
      toggleCamera={toggleCamera}
      captureImage={captureImage}
      handleFileChange={handleFileChange}
      selectedFile={selectedFile}
      loading={loading}
      handleSubmit={handleSubmit}
    />
    
    <ImagePreview 
      cameraActive={cameraActive}
      videoRef={videoRef}
      previewUrl={previewUrl}
    />
    
    <PredictionDisplay 
      prediction={prediction} 
      error={error} 
    />
  </div>
);

export default MainContent;