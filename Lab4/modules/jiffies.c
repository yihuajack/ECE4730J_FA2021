/* simple_module.c - a simple template for a loadable kernel module in Linux,
   based on the hello world kernel module example on pages 338-339 of Robert
   Love's "Linux Kernel Development, Third Edition."
 */

#include <linux/init.h>
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/jiffies.h>

/* init function - logs that initialization happened, returns success */
static int 
simple_init(void)
{
    printk(KERN_ALERT "simple module initialized\n");
    printk(KERN_INFO "jiffies = %lu\n", jiffies);
    return 0;
}

/* exit function - logs that the module is being removed */
static void 
simple_exit(void)
{
    printk(KERN_INFO "jiffies = %lu\n", jiffies);
    printk(KERN_ALERT "simple module is being unloaded\n");
}

module_init(simple_init);
module_exit(simple_exit);

MODULE_LICENSE ("GPL");
MODULE_AUTHOR ("Yihua Liu");
MODULE_DESCRIPTION ("The Kernel Module to Get Jiffies for ECE4730J Lab4");
