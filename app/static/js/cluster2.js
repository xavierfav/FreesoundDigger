var slyelement_list = []
var frame_ist = []
var slidee_list = []
var wrap_list = []
var page_list = []

function create_embed(sound_id) {
    return [
        '<li><div style="text-align:center"><iframe frameborder="0" scrolling="no" src="https://www.freesound.org/embed/sound/iframe/' + sound_id + '/simple/medium/" width="481" height="86"></iframe></div></li>'      
    ]   
}   

function append_embed(embed, cluster_id) {
    slidee_list[cluster_id].append(embed);
	//$('#slidee_'+cluster_id).append(embed);
    // here target the right container with cluster_id arg
}
   
function add_embeds(list_ids, cluster_id) {
    for (item=0; item < list_ids.length; item++) {
        append_embed(create_embed(list_ids[item]), cluster_id)
    }
}

function create_sly_container(cluster_id) {
	return [
		'<div class="wrap"><div style="height: 120px;" id ="frame_' + cluster_id + '" class="frame"><ul id="slidee_' + cluster_id + '" class="slidee"></ul></div> <div id="scrollbar_' + cluster_id + '" class="scrollbar"> <div class="handle"></div></div></div>'
	]
}

function add_sly(cluster_id) {
	$('#cluster_container').append(create_sly_container(cluster_id))
	add_sly_obj(cluster_id, [])
}


var STATE = 'open'
function change_state_to_close(){
    STATE = 'close'
}
function change_state_to_open() {
    STATE = 'open'
}

function close_gate() {
    change_state_to_close();
    setTimeout(change_state_to_open, 500)
    // care with this timer if the server is slower than 0.5sec to respond...
}


function request_sound_ids(page, cluster_id) {
    $.getJSON($SCRIPT_ROOT + '/_get_sound_id', {
        page: page,
        cluster_id: cluster_id,  
    }, function(data) {
        //console.log(slyelement.obj.pos.dest)
        console.log(data.list_ids)
        add_embeds(data.list_ids, cluster_id)
        //console.log(data.list_ids)
        slyelement_list[cluster_id].obj.reload()
        });
    return false;   
}


function add_sly_obj(cluster_id, sound_ids) {
	var $frame  = $('#frame_'+cluster_id);
	var $slidee = $frame.children('ul').eq(0);
	var $wrap   = $frame.parent();	
    var slyelement = {
        obj: {},
        el: $frame,//y'.frame',
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
            scrollBar: $wrap.find('.scrollbar')//'#scrollbar_'+cluster_id//$('.scrollbar')
        }
    };
    slyelement.obj = new Sly($(slyelement.el), slyelement.options);
    slyelement.obj.on('move cycle', function () {
    if (this.pos.dest > this.pos.end - 40 && this.pos.dest > 0 && STATE == 'open' ) {
            close_gate()
            // here request the items for the list
            request_sound_ids(page_list[cluster_id],cluster_id);
            page_list[cluster_id] += 1 
			this.reload();           
		}
	});    
    slyelement.obj.init();
    slyelement_list.push(slyelement)
	frame_ist.push($frame)
	slidee_list.push($slidee)
	wrap_list.push($wrap)
    page_list.push(0)
}

function cluster_result(ids_in_clusters) {
     for (cluster_id=0; cluster_id < ids_in_clusters.length; cluster_id++) {
         // add all sly elements
         //console.log(cluster_id)
         //console.log(ids_in_clusters[cluster_id])
         add_sly(cluster_id)//, ids_in_clusters[cluster_id])
         add_embeds(ids_in_clusters[cluster_id], cluster_id)
         slyelement_list[cluster_id].obj.reload()
         page_list[cluster_id] += 1 
     }
}

$(function(){
	var submit_form = function(e) {
    $.getJSON($SCRIPT_ROOT + '/_cluster', {
      query: $('input[name="query"]').val(),
    }, function(data) {
        //console.log(data.result)
		cluster_result(data.result)
		$('input[name=query]').focus().select();
    });
    return false;
  };
  $('a#calculate').bind('click', submit_form);
  $('input[type=text]').bind('keydown', function(e) {
    if (e.keyCode == 13) {
      submit_form(e);
    }
  });
  $('input[name=query]').focus();
});




///////////////////////////////////////////////////////////
//$(function(){
//    add_sly_obj(0,0)
//    
//    var submit_form = function(e) {
//    $.getJSON($SCRIPT_ROOT + '/_cluster', {
//      query: $('input[name="query"]').val(),
//    }, function(data) {
//    $('#clusters').text(data.result);
//        tags = data.result
//        $('.cluster_div').remove()
//        for (id_cluster=0; id_cluster < tags.length; id_cluster++) {
//            (function () {
//                var element = document.createElement("div");
//                element.className = 'cluster_div'
//                element.id = 'c'+id_cluster
//                //element.appendChild(document.createTextNode(''));
//                document.getElementById('cloudContainer').appendChild(element);
//                createTagCloud(tags[id_cluster], 'c'+id_cluster)
//                //console.log('div ' + id_cluster.toString())
//                var id = id_cluster.toString()
//                document.getElementById('c'+id_cluster).addEventListener('click', function(){
//                    console.log('click on div ' + id)
//                    send_cluster(id);
//                }, false)  
//            }());
//        }
//      $('input[name=query]').focus().select();
//    });
//    return false;
//  };
//  $('a#calculate').bind('click', submit_form);
//  $('input[type=text]').bind('keydown', function(e) {
//    if (e.keyCode == 13) {
//      submit_form(e);
//    }
//  });
//  $('input[name=query]').focus();
//});

//$(window).resize(function(e) {
//  slyelement.obj.reload();
//});


//
//jQuery(function($){
//	'use strict';
//
//	// -------------------------------------------------------------
//	//   Basic Navigation
//	// -------------------------------------------------------------
//	(function () {
//		var $frame  = $('#basic');
//		var $slidee = $frame.children('ul').eq(0);
//		var $wrap   = $frame.parent();
//
//		// Call Sly on frame
//		$frame.sly({
//			horizontal: 1,
//			itemNav: 'basic',
//			smart: 1,
//			activateOn: 'click',
//			mouseDragging: 1,
//			touchDragging: 1,
//			releaseSwing: 1,
//			startAt: 3,
//			scrollBar: $wrap.find('.scrollbar'),
//			scrollBy: 1,
//			pagesBar: $wrap.find('.pages'),
//			activatePageOn: 'click',
//			speed: 300,
//			elasticBounds: 1,
//			easing: 'easeOutExpo',
//			dragHandle: 1,
//			dynamicHandle: 1,
//			clickBar: 1,
//
//			// Buttons
//			forward: $wrap.find('.forward'),
//			backward: $wrap.find('.backward'),
//			prev: $wrap.find('.prev'),
//			next: $wrap.find('.next'),
//			prevPage: $wrap.find('.prevPage'),
//			nextPage: $wrap.find('.nextPage')
//		});
//
//		// To Start button
//		$wrap.find('.toStart').on('click', function () {
//			var item = $(this).data('item');
//			// Animate a particular item to the start of the frame.
//			// If no item is provided, the whole content will be animated.
//			$frame.sly('toStart', item);
//		});
//
//		// To Center button
//		$wrap.find('.toCenter').on('click', function () {
//			var item = $(this).data('item');
//			// Animate a particular item to the center of the frame.
//			// If no item is provided, the whole content will be animated.
//			$frame.sly('toCenter', item);
//		});
//
//		// To End button
//		$wrap.find('.toEnd').on('click', function () {
//			var item = $(this).data('item');
//			// Animate a particular item to the end of the frame.
//			// If no item is provided, the whole content will be animated.
//			$frame.sly('toEnd', item);
//		});
//
//		// Add item
//		$wrap.find('.add').on('click', function () {
//			$frame.sly('add', '<li>' + $slidee.children().length + '</li>');
//		});
//
//		// Remove item
//		$wrap.find('.remove').on('click', function () {
//			$frame.sly('remove', -1);
//		});
//	}());
//
//	// -------------------------------------------------------------
//	//   Centered Navigation
//	// -------------------------------------------------------------
//	(function () {
//		var $frame = $('#centered');
//		var $wrap  = $frame.parent();
//
//		// Call Sly on frame
//		$frame.sly({
//			horizontal: 1,
//			itemNav: 'centered',
//			smart: 1,
//			activateOn: 'click',
//			mouseDragging: 1,
//			touchDragging: 1,
//			releaseSwing: 1,
//			startAt: 4,
//			scrollBar: $wrap.find('.scrollbar'),
//			scrollBy: 1,
//			speed: 300,
//			elasticBounds: 1,
//			easing: 'easeOutExpo',
//			dragHandle: 1,
//			dynamicHandle: 1,
//			clickBar: 1,
//
//			// Buttons
//			prev: $wrap.find('.prev'),
//			next: $wrap.find('.next')
//		});
//	}());
//
//	// -------------------------------------------------------------
//	//   Force Centered Navigation
//	// -------------------------------------------------------------
//	(function () {
//		var $frame = $('#forcecentered');
//		var $wrap  = $frame.parent();
//
//		// Call Sly on frame
//		$frame.sly({
//			horizontal: 1,
//			itemNav: 'forceCentered',
//			smart: 1,
//			activateMiddle: 1,
//			activateOn: 'click',
//			mouseDragging: 1,
//			touchDragging: 1,
//			releaseSwing: 1,
//			startAt: 0,
//			scrollBar: $wrap.find('.scrollbar'),
//			scrollBy: 1,
//			speed: 300,
//			elasticBounds: 1,
//			easing: 'easeOutExpo',
//			dragHandle: 1,
//			dynamicHandle: 1,
//			clickBar: 1,
//
//			// Buttons
//			prev: $wrap.find('.prev'),
//			next: $wrap.find('.next')
//		});
//	}());
//
//	// -------------------------------------------------------------
//	//   Cycle By Items
//	// -------------------------------------------------------------
//	(function () {
//		var $frame = $('#cycleitems');
//		var $wrap  = $frame.parent();
//
//		// Call Sly on frame
//		$frame.sly({
//			horizontal: 1,
//			itemNav: 'basic',
//			smart: 1,
//			activateOn: 'click',
//			mouseDragging: 1,
//			touchDragging: 1,
//			releaseSwing: 1,
//			startAt: 0,
//			scrollBar: $wrap.find('.scrollbar'),
//			scrollBy: 1,
//			speed: 300,
//			elasticBounds: 1,
//			easing: 'easeOutExpo',
//			dragHandle: 1,
//			dynamicHandle: 1,
//			clickBar: 1,
//
//			// Cycling
//			cycleBy: 'items',
//			cycleInterval: 1000,
//			pauseOnHover: 1,
//
//			// Buttons
//			prev: $wrap.find('.prev'),
//			next: $wrap.find('.next')
//		});
//
//		// Pause button
//		$wrap.find('.pause').on('click', function () {
//			$frame.sly('pause');
//		});
//
//		// Resume button
//		$wrap.find('.resume').on('click', function () {
//			$frame.sly('resume');
//		});
//
//		// Toggle button
//		$wrap.find('.toggle').on('click', function () {
//			$frame.sly('toggle');
//		});
//	}());
//
//	// -------------------------------------------------------------
//	//   Cycle By Pages
//	// -------------------------------------------------------------
//	(function () {
//		var $frame = $('#cyclepages');
//		var $wrap  = $frame.parent();
//
//		// Call Sly on frame
//		$frame.sly({
//			horizontal: 1,
//			itemNav: 'basic',
//			smart: 1,
//			activateOn: 'click',
//			mouseDragging: 1,
//			touchDragging: 1,
//			releaseSwing: 1,
//			startAt: 0,
//			scrollBar: $wrap.find('.scrollbar'),
//			scrollBy: 1,
//			pagesBar: $wrap.find('.pages'),
//			activatePageOn: 'click',
//			speed: 300,
//			elasticBounds: 1,
//			easing: 'easeOutExpo',
//			dragHandle: 1,
//			dynamicHandle: 1,
//			clickBar: 1,
//
//			// Cycling
//			cycleBy: 'pages',
//			cycleInterval: 1000,
//			pauseOnHover: 1,
//			startPaused: 1,
//
//			// Buttons
//			prevPage: $wrap.find('.prevPage'),
//			nextPage: $wrap.find('.nextPage')
//		});
//
//		// Pause button
//		$wrap.find('.pause').on('click', function () {
//			$frame.sly('pause');
//		});
//
//		// Resume button
//		$wrap.find('.resume').on('click', function () {
//			$frame.sly('resume');
//		});
//
//		// Toggle button
//		$wrap.find('.toggle').on('click', function () {
//			$frame.sly('toggle');
//		});
//	}());
//
//	// -------------------------------------------------------------
//	//   One Item Per Frame
//	// -------------------------------------------------------------
//	(function () {
//		var $frame = $('#oneperframe');
//		var $wrap  = $frame.parent();
//
//		// Call Sly on frame
//		$frame.sly({
//			horizontal: 1,
//			itemNav: 'forceCentered',
//			smart: 1,
//			activateMiddle: 1,
//			mouseDragging: 1,
//			touchDragging: 1,
//			releaseSwing: 1,
//			startAt: 0,
//			scrollBar: $wrap.find('.scrollbar'),
//			scrollBy: 1,
//			speed: 300,
//			elasticBounds: 1,
//			easing: 'easeOutExpo',
//			dragHandle: 1,
//			dynamicHandle: 1,
//			clickBar: 1,
//
//			// Buttons
//			prev: $wrap.find('.prev'),
//			next: $wrap.find('.next')
//		});
//	}());
//
//	// -------------------------------------------------------------
//	//   Crazy
//	// -------------------------------------------------------------
//	(function () {
//		var $frame  = $('#crazy');
//		var $slidee = $frame.children('ul').eq(0);
//		var $wrap   = $frame.parent();
//
//		// Call Sly on frame
//		$frame.sly({
//			horizontal: 1,
//			itemNav: 'basic',
//			smart: 1,
//			activateOn: 'click',
//			mouseDragging: 1,
//			touchDragging: 1,
//			releaseSwing: 1,
//			startAt: 3,
//			scrollBar: $wrap.find('.scrollbar'),
//			scrollBy: 1,
//			pagesBar: $wrap.find('.pages'),
//			activatePageOn: 'click',
//			speed: 300,
//			elasticBounds: 1,
//			easing: 'easeOutExpo',
//			dragHandle: 1,
//			dynamicHandle: 1,
//			clickBar: 1,
//
//			// Buttons
//			forward: $wrap.find('.forward'),
//			backward: $wrap.find('.backward'),
//			prev: $wrap.find('.prev'),
//			next: $wrap.find('.next'),
//			prevPage: $wrap.find('.prevPage'),
//			nextPage: $wrap.find('.nextPage')
//		});
//
//		// To Start button
//		$wrap.find('.toStart').on('click', function () {
//			var item = $(this).data('item');
//			// Animate a particular item to the start of the frame.
//			// If no item is provided, the whole content will be animated.
//			$frame.sly('toStart', item);
//		});
//
//		// To Center button
//		$wrap.find('.toCenter').on('click', function () {
//			var item = $(this).data('item');
//			// Animate a particular item to the center of the frame.
//			// If no item is provided, the whole content will be animated.
//			$frame.sly('toCenter', item);
//		});
//
//		// To End button
//		$wrap.find('.toEnd').on('click', function () {
//			var item = $(this).data('item');
//			// Animate a particular item to the end of the frame.
//			// If no item is provided, the whole content will be animated.
//			$frame.sly('toEnd', item);
//		});
//
//		// Add item
//		$wrap.find('.add').on('click', function () {
//			$frame.sly('add', '<li>' + $slidee.children().length + '</li>');
//		});
//
//		// Remove item
//		$wrap.find('.remove').on('click', function () {
//			$frame.sly('remove', -1);
//		});
//	}());
//});
