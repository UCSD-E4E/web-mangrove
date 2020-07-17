from app import app 
from app import server
  
if __name__ == "__main__": 
        # app.run() 
        app.run_server(debug=True)
        server.run(debug=True)