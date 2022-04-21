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
    multiple: false,
    formatsAllowed: ".jpg,.mp4",
    maxSize: 20,
    uploadAPI:  {
      url:"https://example-file-upload-api",
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

  chooseVideo()
  {

  }

  uploadVideo()
  {
    this.isUploading = true;
  }

}
