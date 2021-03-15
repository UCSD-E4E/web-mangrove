import React from 'react';
import { io } from 'socket.io-client';
import axios from 'axios';
import { Progress } from 'antd';
import styles from './index.module.scss';

const Classify: React.FC = () => {
  const [socketId, setSocketId] = React.useState<string>();
  const [stage, setStage] = React.useState<string>();
  const [progress, setProgress] = React.useState<number>();

  React.useEffect(() => {
    const socket = io('http://localhost:5000');
    socket.on('connect', () => {
      setSocketId(socket.id);
    });
    socket.on('message', (data: string) => {
      const [progressType, progressNumber] = data.split(':');
      setStage(progressType);
      setProgress(parseInt(progressNumber, 10));
    });
  }, []);

  return (
    <div>
      <h1>Classify</h1>
      <p>
        After uploading the file and all tiles exist, click below to leverage machine learning to
        classify the drone imagery.
      </p>
      <button
        type="submit"
        onClick={() => {
          axios.post('http://localhost:5000/files/retile', { room: socketId }).catch(() => {});
        }}
      >
        Classify
      </button>
      <Progress percent={progress} className={styles.progress} />
      <span>{stage}</span>
    </div>
  );
};

export default Classify;
