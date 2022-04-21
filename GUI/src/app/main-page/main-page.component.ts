import { Component, OnInit } from '@angular/core';

@Component({
  selector: 'app-main-page',
  templateUrl: './main-page.component.html',
  styleUrls: ['./main-page.component.css']
})
export class MainPageComponent implements OnInit {
  isUploading: boolean = false; 
  isRecieved: boolean = false;
  hasSelectedVideo = false; 

  afuConfig = {
    uploadAPI: {
      url:"https://example-file-upload-api"
    }
};

  constructor() { }

  ngOnInit(): void {
  }

  chooseVideo()
  {

  }

  uploadVideo()
  {
    this.isUploading = true;
  }

}
