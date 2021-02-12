import React from 'react';
import { io } from 'socket.io-client';
import axios from 'axios';

const Classify: React.FC = () => {
  const [socketId, setSocketId] = React.useState<string>();

  React.useEffect(() => {
    const socket = io('http://localhost:5000');
    socket.on('connect', () => {
      setSocketId(socket.id);
    });
    socket.on('message', (data: any) => {
      const progress = document.getElementById('retile-progress')!;
      progress.innerHTML = `Progress: ${data}`;
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
        Retile
      </button>
      <span id="retile-progress">Progress: </span>
    </div>
  );
};

export default Classify;
