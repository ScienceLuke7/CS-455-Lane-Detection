import { Component, OnInit } from '@angular/core';

@Component({
  selector: 'app-main-page',
  templateUrl: './main-page.component.html',
  styleUrls: ['./main-page.component.css']
})
export class MainPageComponent implements OnInit {
  isUploading: boolean = false; 
  isVideo: boolean = true;
  showResults: boolean = false;

  afuConfig = {
    multiple: false,
    formatsAllowed: ".jpg,.mp4",
    maxSize: 20,
    uploadAPI:  {
      url:"http://127.0.0.1:5000/postRawVideo",
    },
    hideProgressBar: false,
    hideResetBtn: true,
    hideSelectBtn: false,
    fileNameIndex: true,
    autoUpload: false,
    replaceTexts: {
      selectFileBtn: 'Select Files',
      resetBtn: 'Reset',
      uploadBtn: 'Upload',
      attachPinBtn: 'Attach Files...',
      afterUploadMsg_success: 'Successfully Uploaded !',
      sizeLimit: 'Size Limit'
    }
  };

  constructor() { }

  ngOnInit(): void {
    this.showResults = false;

  }

  showResult()
  {
    this.showResults = true;
  }

}
