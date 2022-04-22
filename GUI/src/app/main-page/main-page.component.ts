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
  isVideo: boolean = true;

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
      afterUploadMsg_error: 'Upload Failed !',
      sizeLimit: 'Size Limit'
    }
  };

  afuConfig2 = {
    multiple: false,
    formatsAllowed: ".jpg,.mp4",
    maxSize: 20,
    uploadAPI:  {
      url:"http://127.0.0.1:5000/postRawImage",
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
      afterUploadMsg_error: 'Upload Failed !',
      sizeLimit: 'Size Limit'
    }
  };

  constructor() { }

  ngOnInit(): void {
  }

  uploadVideo()
  {
    this.isUploading = true;
  }

  displayResults()
  {

  }

}
