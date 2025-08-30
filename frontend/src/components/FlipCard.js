import React from 'react';

const FlipCard = ({ imageSrc, altText, className, backContent, isCard }) => (
  <div className={`flip-card ${isCard ? 'side-card' : ''}`}>
    <div className="flip-card-inner">
      <div className="flip-card-front">
        <img src={imageSrc} alt={altText} className={className} />
      </div>
      <div className="flip-card-back">
        <p>{backContent}</p>
      </div>
    </div>
  </div>
);

export default FlipCard;