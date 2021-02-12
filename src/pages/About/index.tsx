import React from 'react';
import Cover from '../../components/Cover';
import styles from './index.module.scss';

const About: React.FC = () => {
  return (
    <>
      <Cover height="40vh" isHome={false} />
      <div className={styles.about}>
        <h1>Why Mangroves?</h1>
        <p>
          Mangrove forests are highly productive ecosystems and play a vital role in carbon
          sequestration. Accurately monitoring their growth in an automated way has presented a
          challenge for many conservation groups, especially those with smaller budgets who may not
          have the financial resources to deploy satellites or expensive drones with multispectral
          sensors. Drones with cheaper sensors have the advantage of accessibility, but a
          streamlined workflow for drone imagery classification is still lacking.
        </p>
        <p>
          This image classification tool is an accessible and automated workflow for conservation
          groups to quantify the amount of mangroves within their site by allowing them to upload
          cheaply acquired high resolution imagery (drone imagery) to a website and quantify the
          amount of mangroves in their site using a CNN.
        </p>
      </div>
    </>
  );
};

export default About;
