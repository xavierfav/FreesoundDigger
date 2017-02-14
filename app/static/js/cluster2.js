var slyelement = {
  obj: {},
  el: '.frame',
  options: {
    horizontal: 1,
    itemNav: 'forceCentered',
    smart: 1,
    activateMiddle: 1,
    activateOn: 'click',
    mouseDragging: 1,
    touchDragging: 1,
    releaseSwing: 1,
    startAt: 0,
    scrollBy: 1,
    speed: 300,
    elasticBounds: 1,
    easing: 'swing', // easeInOutElastic, easeOutBounce
    scrollBar: $('.scrollbar')
  }
};

function create_embed(sound_id) {
    return [
        '<li><div style="text-align:center"><iframe frameborder="0" scrolling="no" src="https://www.freesound.org/embed/sound/iframe/' + sound_id + '/simple/medium/" width="481" height="86"></iframe></div></li>'      
    ]   
}   
    
function append_embed(embed, cluster_id) {
    $(".slidee").append(embed);
    // here target the right container with cluster_id arg
}
   
function add_embeds(list_ids, cluster_id) {
    for (item=0; item < list_ids.length; item++) {
        append_embed(create_embed(list_ids[item]), cluster_id)
    }
}

var STATE = 'open'
function change_state_to_close(){
    STATE = 'close'
}
function change_state_to_open() {
    STATE = 'open'
}

function close() {
    change_state_to_close();
    setTimeout(change_state_to_open, 500)
    // care with this timer if the server is slower than 0.5sec to respond...
}


function request_sound_ids(page, cluster_id) {
    $.getJSON($SCRIPT_ROOT + '/_get_sound_id', {
        page: page,
        cluster_id: cluster_id,  
    }, function(data) {
        add_embeds(data.list_ids, cluster_id)
        console.log(data.list_ids)
        slyelement.obj.reload()
        });
    return false;   
}



$(function(){
  slyelement.obj = new Sly($(slyelement.el), slyelement.options);
  slyelement.obj.on('move cycle', function () {
		if (this.pos.dest > this.pos.end - 40 && STATE == 'open' ) {
            close()
            request_sound_ids(1,2);
			this.reload();
            console.log('reload')
            
		}
	});
  slyelement.obj.init();
});

$(window).resize(function(e) {
  slyelement.obj.reload();
});