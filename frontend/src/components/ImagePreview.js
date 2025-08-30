import React from 'react';

const ImagePreview = ({ cameraActive, videoRef, previewUrl }) => (
  <div className="preview-container">
    {cameraActive ? (
      <video 
        ref={videoRef}
        autoPlay
        playsInline
        className="video-preview"
      />
    ) : previewUrl ? (
      <img
        src={previewUrl}
        alt="preview"
        className="preview-image"
      />
    ) : (
      <div className="empty-preview">
        <p>No image selected</p>
      </div>
    )}
  </div>
);

export default ImagePreview;