import React from 'react';
import { io } from 'socket.io-client';
import axios from 'axios';

const App: React.FC = () => {
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
    <>
      <input type="file" accept=".tif,.tiff,.zip" id="fileinput" />
      <button
        type="submit"
        onClick={() => {
          const fileInput = document.getElementById('fileinput') as any;
          const file = fileInput.files[0];
          const extension = fileInput.value.split('.').pop();
          axios.post('http://localhost:5000/files/upload', file, {
            headers: {
              'Content-Type': extension === 'zip' ? 'application/zip' : 'image/tiff',
            },
            onUploadProgress: (event) => {
              const progress = document.getElementById('upload-progress')!;
              progress.innerHTML = `Progress: ${Math.round((event.loaded * 100) / event.total)}%`;
            },
          });
        }}
      >
        Upload
      </button>
      <span id="upload-progress">Progress: </span>
      <button
        type="submit"
        onClick={() => {
          axios.post('http://localhost:5000/files/retile', { room: socketId }).catch(() => {});
        }}
      >
        Retile
      </button>
      <span id="retile-progress">Progress: </span>
    </>
  );
};

export default App;
