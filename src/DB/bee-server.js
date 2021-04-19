
var http = require('http'); // Import Node.js core module
var url = require('url'); //Not used but maybe later
var fs = require('fs'); //Import filestytem module

var server = http.createServer(function (req, res) {   //create web server
  if (req.url == '/') { //check the URL of the current request

    // set response header
    res.writeHead(200, { 'Content-Type': 'text/html' });
    console.log("ay");

    // set response content    
    res.write('<html><body><p>This is the test home Page.</p></body></html>');
    res.end();

  }
  else if (req.url == "/test") {

    res.writeHead(200, { 'Content-Type': 'text/html' });
    res.write('<html><body><p>This is the test endpoint.</p></body></html>');
    res.end();

  }
  else if (req.url == "/admin") {

    res.writeHead(200, { 'Content-Type': 'text/html' });
    res.write('<html><body><p>This is admin Page.</p></body></html>');
    res.end();

  }
  else if (req.url == "/filenames") {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    fs.readdir('./', (err, files) => {
      files.forEach(file => {
        console.log(file);
      });
      var data = JSON.stringify(files);
      res.write(data);
      res.end();
    });
  } 
  else
    res.end('Invalid Request!');

});

server.listen(5000); //6 - listen for any incoming requests

console.log('Node.js web server at port 5000 is running..')
