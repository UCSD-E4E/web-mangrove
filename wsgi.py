from app import app 
from app import server
  
if __name__ == "__main__": 
        # app.run() 
        app.run_server(debug=False)
        server.run(debug=True)