import React from 'react';
import axios from 'axios';
import { Button, Progress } from 'antd';
import styles from './index.module.scss';

const Upload: React.FC = () => {
  const [uploadProgress, setUploadProgress] = React.useState(0);
  return (
    <div className={styles.upload}>
      <h1>File Upload</h1>
      <p>
        Choose .zip file to upload. Click Upload to unzip automatically. Unzipped tiles will be
        displayed on the right.
      </p>
      <input
        type="file"
        accept=".tif,.tiff,.zip"
        id="fileinput"
        onSubmit={() => {
          setUploadProgress(0);
        }}
        className={styles.fileSelector}
      />
      <Button
        type="primary"
        shape="round"
        size="large"
        onClick={() => {
          const fileInput = document.getElementById('fileinput') as any;
          const file = fileInput.files[0];
          const extension = fileInput.value.split('.').pop();
          axios.post('http://localhost:5000/files/upload', file, {
            headers: {
              'Content-Type': extension === 'zip' ? 'application/zip' : 'image/tiff',
            },
            onUploadProgress: (event) => {
              setUploadProgress(Math.round((event.loaded * 100) / event.total));
            },
          });
        }}
      >
        Upload
      </Button>
      <Progress percent={uploadProgress} className={styles.progress} />
    </div>
  );
};

export default Upload;
