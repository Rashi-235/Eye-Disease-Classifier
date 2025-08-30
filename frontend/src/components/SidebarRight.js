import React from 'react';
import FlipCard from './FlipCard';

const SidebarRight = ({ eyeConditions }) => (
  <div className="sidebar right">
    <FlipCard 
      imageSrc="/3.jpeg" 
      altText="Eyelid" 
      className="side-image" 
      backContent={eyeConditions.eyelid} 
      isCard={true}
    />
    
    <FlipCard 
      imageSrc="/6.jpeg" 
      altText="Uveitis" 
      className="side-image" 
      backContent={eyeConditions.uveitis} 
      isCard={true}
    />
  </div>
);

export default SidebarRight;