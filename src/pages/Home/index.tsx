import React from 'react';
import Classify from '../../components/Classify';
import Cover from '../../components/Cover';
import Upload from '../../components/Upload';
import styles from './index.module.scss';

const Home: React.FC = () => {
  return (
    <>
      <Cover height="100vh" isHome={true} />
      <div className={styles.process}>
        <Upload />
        <Classify />
      </div>
    </>
  );
};

export default Home;
