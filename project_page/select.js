var current_3D_SceneList = ['blooming', 'withering', 'sprouts', 'baking', 'candle', 'icecream', 'banana'];
var current_3D_Scene = "blooming";
var current_3D_SceneId = 0;

function Change_TOI_scene(idx) {

  var li_list = document.querySelector("#toi-buttons").children;

  // console.log(idx);
  console.log(li_list);
  for(i = 0; i < li_list.length; i++){
      li_list[i].classList = ['button']
  }
  li_list[idx].classList.add('is-primary');

  current_3D_Scene = current_3D_SceneList[idx];
  current_3D_SceneId = idx;
  let video = document.getElementById("toi-video");
  old_src = video.src;
  new_scr = old_src.split('/');
  new_scr[new_scr.length-1] = current_3D_Scene + "2.mp4";
  new_video_dir = new_scr.join('/');
  video.src = new_video_dir;
  video.load()
}

function Change_comp_scene(idx) {

    var li_list = document.querySelector("#comp-buttons").children;
  
    // console.log(idx);
    console.log(li_list);
    for(i = 0; i < li_list.length; i++){
        li_list[i].classList = ['button']
    }
    li_list[idx].classList.add('is-primary');
  
    current_3D_Scene = current_3D_SceneList[idx];
    current_3D_SceneId = idx;
    let videos = document.querySelectorAll(".comp-videos");
    for(let video of videos){
        old_src = video.src;
        new_scr = old_src.split('/');
        method = new_scr[new_scr.length-1].split('_')[1];
        new_scr[new_scr.length-1] = current_3D_Scene +'_' + method;
        new_video_dir = new_scr.join('/');
        video.src = new_video_dir;
        video.load()
    } 
  }
